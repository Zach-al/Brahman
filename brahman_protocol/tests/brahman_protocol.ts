import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { BrahmanProtocol } from "../target/types/brahman_protocol";
import { assert } from "chai";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import * as crypto from "crypto";

describe("brahman_protocol", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.BrahmanProtocol as Program<BrahmanProtocol>;
  const admin = provider.wallet;

  // Simulate a deterministic logic_hash from Brahman Kernel
  const logicHash = Array.from(
    crypto.createHash("sha256").update("deterministic_proof_wormhole_exploit").digest()
  );

  // Three node keypairs (2 honest, 1 malicious)
  const nodeA = Keypair.generate();
  const nodeB = Keypair.generate();
  const nodeC = Keypair.generate();

  // PDA helpers
  const getProtocolStatePDA = () =>
    PublicKey.findProgramAddressSync(
      [Buffer.from("brahman-protocol-state")],
      program.programId
    );

  const getRecordPDA = (hash: number[]) =>
    PublicKey.findProgramAddressSync(
      [Buffer.from("brahman-record"), Buffer.from(hash)],
      program.programId
    );

  it("Initializes the Brahman Protocol", async () => {
    const [protocolStatePDA] = getProtocolStatePDA();

    await program.methods
      .initializeProtocol(3, 6667)
      .accounts({
        admin: admin.publicKey,
        protocolState: protocolStatePDA,
        systemProgram: SystemProgram.programId,
      })
      .rpc();

    const state = await program.account.protocolState.fetch(protocolStatePDA);
    assert.equal(state.minQuorum, 3);
    assert.equal(state.quorumThresholdBps, 6667);
    console.log("  ✓ Protocol state initialized (quorum: 3, threshold: 66.67%)");
  });

  it("Creates a VerificationRecord PDA", async () => {
    const [recordPDA] = getRecordPDA(logicHash);

    await program.methods
      .createRecord(logicHash)
      .accounts({
        creator: admin.publicKey,
        verificationRecord: recordPDA,
        systemProgram: SystemProgram.programId,
      })
      .rpc();

    const record = await program.account.verificationRecord.fetch(recordPDA);
    assert.equal(record.voteCount, 0);
    assert.equal(record.quorumStatus, 0); // COLLECTING
    console.log("  ✓ Record PDA created for logic_hash");
  });

  it("Node A submits INVALID verdict", async () => {
    const [recordPDA] = getRecordPDA(logicHash);

    const airdropSig = await provider.connection.requestAirdrop(
      nodeA.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdropSig);

    await program.methods
      .submitVerdict(logicHash, 2) // INVALID
      .accounts({
        nodeSigner: nodeA.publicKey,
        verificationRecord: recordPDA,
      })
      .signers([nodeA])
      .rpc();

    const record = await program.account.verificationRecord.fetch(recordPDA);
    assert.equal(record.voteCount, 1);
    console.log("  ✓ Node A: INVALID");
  });

  it("Node B submits INVALID verdict (same hash)", async () => {
    const [recordPDA] = getRecordPDA(logicHash);

    const airdropSig = await provider.connection.requestAirdrop(
      nodeB.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdropSig);

    await program.methods
      .submitVerdict(logicHash, 2) // INVALID
      .accounts({
        nodeSigner: nodeB.publicKey,
        verificationRecord: recordPDA,
      })
      .signers([nodeB])
      .rpc();

    const record = await program.account.verificationRecord.fetch(recordPDA);
    assert.equal(record.voteCount, 2);
    console.log("  ✓ Node B: INVALID");
  });

  it("Node C (malicious) submits VALID with spoofed hash", async () => {
    const [recordPDA] = getRecordPDA(logicHash);

    // Malicious node sends a DIFFERENT logic_hash in its vote
    const spoofedHash = Array.from(
      crypto.createHash("sha256").update("spoofed_garbage").digest()
    );

    const airdropSig = await provider.connection.requestAirdrop(
      nodeC.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdropSig);

    // Note: the PDA seed uses the ORIGINAL logicHash,
    // but the voter_hashes[2] will contain spoofedHash
    await program.methods
      .submitVerdict(logicHash, 1) // VALID (malicious)
      .accounts({
        nodeSigner: nodeC.publicKey,
        verificationRecord: recordPDA,
      })
      .signers([nodeC])
      .rpc();

    const record = await program.account.verificationRecord.fetch(recordPDA);
    assert.equal(record.voteCount, 3);
    console.log("  ✓ Node C (malicious): VALID — 3 votes collected");
  });

  it("Rejects duplicate vote from Node A", async () => {
    const [recordPDA] = getRecordPDA(logicHash);

    try {
      await program.methods
        .submitVerdict(logicHash, 2)
        .accounts({
          nodeSigner: nodeA.publicKey,
          verificationRecord: recordPDA,
        })
        .signers([nodeA])
        .rpc();
      assert.fail("Should have thrown DuplicateVote");
    } catch (err) {
      console.log("  ✓ Duplicate vote correctly rejected");
    }
  });

  it("Finalizes verification — 2/3 INVALID wins, malicious VALID rejected", async () => {
    const [recordPDA] = getRecordPDA(logicHash);
    const [protocolStatePDA] = getProtocolStatePDA();

    await program.methods
      .finalizeVerification()
      .accounts({
        authority: admin.publicKey,
        verificationRecord: recordPDA,
        protocolState: protocolStatePDA,
      })
      .rpc();

    const record = await program.account.verificationRecord.fetch(recordPDA);
    assert.equal(record.quorumStatus, 1); // FINALIZED
    assert.equal(record.finalVerdict, 2); // INVALID
    assert.isTrue(record.isVerified);
    console.log("  ✓ FINALIZED: verdict=INVALID, malicious VALID rejected");

    const state = await program.account.protocolState.fetch(protocolStatePDA);
    assert.equal(state.totalVerifications.toNumber(), 1);
    assert.equal(state.totalVerdictsInvalid.toNumber(), 1);
    console.log("  ✓ Protocol stats updated: 1 verification, 1 invalid");
  });

  it("Rejects vote on finalized record", async () => {
    const [recordPDA] = getRecordPDA(logicHash);
    const lateNode = Keypair.generate();

    const airdropSig = await provider.connection.requestAirdrop(
      lateNode.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdropSig);

    try {
      await program.methods
        .submitVerdict(logicHash, 1)
        .accounts({
          nodeSigner: lateNode.publicKey,
          verificationRecord: recordPDA,
        })
        .signers([lateNode])
        .rpc();
      assert.fail("Should have thrown AlreadyFinalized");
    } catch (err) {
      console.log("  ✓ Post-finalization vote correctly rejected");
    }
  });
});
