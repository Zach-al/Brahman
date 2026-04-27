// Licensed under BSL 1.1 — commercial use requires written permission
// Change date: 2027-01-01 to MIT License by Bhupen Nayak
// Contact: askzachn@gmail.com

use anchor_lang::prelude::*;

declare_id!("hkCmPnS4SRfniSuXhP9yyW55q1fj8xVEyeRcDzSRh6t");

/// Authorized deployer — only this pubkey can call initialize_protocol.
/// DEPLOYMENT CHECKLIST: Verify this matches your deployer wallet before each deploy.
/// Current: Bhupen's dev wallet (EBZFNFHD3riawGyPgVm9fVNGE8aDGoEcW77jrRHdRCz9)
const AUTHORIZED_DEPLOYER: Pubkey = pubkey!("EBZFNFHD3riawGyPgVm9fVNGE8aDGoEcW77jrRHdRCz9");

/// Brahman Verification Protocol — On-Chain Consensus Anchor
///
/// Records off-chain Brahman Kernel verdicts on the Solana blockchain.
///
/// Flow:
///   1. Sovereign Nodes verify a transaction off-chain
///   2. Coordinator calls `create_record` to initialize the PDA
///   3. Each node calls `submit_verdict` to record their vote
///   4. Once quorum is met, coordinator calls `finalize_verification`
///   5. The VerificationRecord PDA is sealed and publicly queryable

#[program]
pub mod brahman_protocol {
    use super::*;

    /// Initialize the global protocol state. Called once by the admin.
    pub fn initialize_protocol(
        ctx: Context<InitializeProtocol>,
        min_quorum: u8,
        quorum_threshold_bps: u16,
    ) -> Result<()> {
        // SECURITY: Only the authorized deployer can initialize
        require!(
            ctx.accounts.admin.key() == AUTHORIZED_DEPLOYER,
            BrahmanError::UnauthorizedInitialization
        );

        let state = &mut ctx.accounts.protocol_state;
        state.admin = ctx.accounts.admin.key();
        state.min_quorum = min_quorum;
        state.quorum_threshold_bps = quorum_threshold_bps;
        state.total_verifications = 0;
        state.total_verdicts_valid = 0;
        state.total_verdicts_invalid = 0;
        state.total_verdicts_disputed = 0;
        state.registered_node_count = 0;

        msg!("Brahman Protocol initialized. Quorum: {}/10000 bps", quorum_threshold_bps);
        Ok(())
    }

    /// Register a node as an authorized validator. Admin only.
    pub fn register_node(
        ctx: Context<AdminAction>,
        node_pubkey: Pubkey,
    ) -> Result<()> {
        let state = &mut ctx.accounts.protocol_state;
        require!(
            ctx.accounts.admin.key() == state.admin,
            BrahmanError::Unauthorized
        );
        let idx = state.registered_node_count as usize;
        require!(idx < 32, BrahmanError::MaxNodesReached);
        // Prevent duplicate registration
        for i in 0..idx {
            require!(state.registered_nodes[i] != node_pubkey, BrahmanError::DuplicateNode);
        }
        state.registered_nodes[idx] = node_pubkey;
        state.registered_node_count += 1;
        msg!("Node registered: {}", node_pubkey);
        Ok(())
    }

    /// Create a new VerificationRecord PDA for a given logic_hash.
    pub fn create_record(
        ctx: Context<CreateRecord>,
        logic_hash: [u8; 32],
    ) -> Result<()> {
        let record = &mut ctx.accounts.verification_record;
        let clock = Clock::get()?;

        record.logic_hash = logic_hash;
        record.created_at = clock.unix_timestamp;
        record.quorum_status = 0; // COLLECTING
        record.final_verdict = 0; // PENDING
        record.is_verified = false;
        record.vote_count = 0;
        record.finalized_at = 0;
        record.authority = Pubkey::default();
        record.final_logic_hash = [0u8; 32];

        msg!("VerificationRecord created for hash: {:?}", &logic_hash[..8]);
        Ok(())
    }

    /// Submit a single node's verdict for a pending transaction.
    pub fn submit_verdict(
        ctx: Context<SubmitVerdict>,
        logic_hash: [u8; 32],
        verdict: u8,
    ) -> Result<()> {
        let record = &mut ctx.accounts.verification_record;
        let state = &ctx.accounts.protocol_state;

        // SECURITY: Only registered nodes can submit verdicts (anti-Sybil)
        let signer = ctx.accounts.node_signer.key();
        let node_count = state.registered_node_count as usize;
        let mut is_registered = false;
        for i in 0..node_count {
            if state.registered_nodes[i] == signer {
                is_registered = true;
                break;
            }
        }
        require!(is_registered, BrahmanError::UnauthorizedNode);

        // Reject if already finalized
        require!(record.quorum_status != 1, BrahmanError::AlreadyFinalized);

        // Reject if max votes reached
        let idx = record.vote_count as usize;
        require!(idx < 10, BrahmanError::MaxVotesReached);

        // Reject duplicate votes from same node
        for i in 0..idx {
            require!(record.voter_pubkeys[i] != signer, BrahmanError::DuplicateVote);
        }

        // Record the vote
        record.voter_pubkeys[idx] = signer;
        record.voter_verdicts[idx] = verdict;
        record.voter_hashes[idx] = logic_hash;
        record.vote_count += 1;

        msg!(
            "Vote #{} from {} — verdict: {}",
            record.vote_count,
            signer,
            verdict
        );

        Ok(())
    }

    /// Finalize verification once quorum votes are collected.
    /// Checks hash agreement and seals the record permanently.
    pub fn finalize_verification(
        ctx: Context<FinalizeVerification>,
    ) -> Result<()> {
        let record = &mut ctx.accounts.verification_record;
        let state = &mut ctx.accounts.protocol_state;
        let clock = Clock::get()?;

        // SECURITY: Only admin can finalize verification records
        require!(
            ctx.accounts.authority.key() == state.admin,
            BrahmanError::UnauthorizedFinalization
        );

        // Must not be already finalized
        require!(record.quorum_status != 1, BrahmanError::AlreadyFinalized);

        // Must have enough votes
        require!(
            record.vote_count >= state.min_quorum,
            BrahmanError::InsufficientVotes
        );

        // Count hash agreements — find the most common hash
        let total = record.vote_count as usize;
        let threshold = ((total as u64) * (state.quorum_threshold_bps as u64)) / 10000;

        let mut best_count: u64 = 0;
        let mut best_idx: usize = 0;

        for i in 0..total {
            let mut count: u64 = 0;
            for j in 0..total {
                if record.voter_hashes[i] == record.voter_hashes[j] {
                    count += 1;
                }
            }
            if count > best_count {
                best_count = count;
                best_idx = i;
            }
        }

        // Check if threshold is met
        if best_count >= threshold {
            record.quorum_status = 1; // FINALIZED
            record.final_verdict = record.voter_verdicts[best_idx];
            record.final_logic_hash = record.voter_hashes[best_idx];
            record.is_verified = true;
            record.finalized_at = clock.unix_timestamp;
            record.authority = ctx.accounts.authority.key();

            state.total_verifications += 1;
            match record.final_verdict {
                1 => state.total_verdicts_valid += 1,
                2 => state.total_verdicts_invalid += 1,
                _ => {}
            }

            msg!("✓ FINALIZED — verdict: {}, votes: {}/{}", 
                 record.final_verdict, best_count, total);
        } else {
            record.quorum_status = 2; // DISPUTED
            record.final_verdict = 4; // DISPUTED
            record.finalized_at = clock.unix_timestamp;
            record.is_verified = false;

            state.total_verifications += 1;
            state.total_verdicts_disputed += 1;

            msg!("✗ DISPUTED — no hash reached {}/{} threshold", threshold, total);
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
// Account Contexts
// ─────────────────────────────────────────────────────────────

#[derive(Accounts)]
pub struct InitializeProtocol<'info> {
    #[account(mut)]
    pub admin: Signer<'info>,

    #[account(
        init,
        payer = admin,
        space = ProtocolState::SPACE,
        seeds = [b"brahman-protocol-state"],
        bump,
    )]
    pub protocol_state: Account<'info, ProtocolState>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(logic_hash: [u8; 32])]
pub struct CreateRecord<'info> {
    #[account(mut)]
    pub creator: Signer<'info>,

    #[account(
        init,
        payer = creator,
        space = VerificationRecord::SPACE,
        seeds = [b"brahman-record", logic_hash.as_ref()],
        bump,
    )]
    pub verification_record: Account<'info, VerificationRecord>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(logic_hash: [u8; 32], verdict: u8)]
pub struct SubmitVerdict<'info> {
    #[account(mut)]
    pub node_signer: Signer<'info>,

    #[account(
        mut,
        seeds = [b"brahman-record", logic_hash.as_ref()],
        bump,
    )]
    pub verification_record: Account<'info, VerificationRecord>,

    #[account(
        seeds = [b"brahman-protocol-state"],
        bump,
    )]
    pub protocol_state: Account<'info, ProtocolState>,
}

#[derive(Accounts)]
pub struct AdminAction<'info> {
    #[account(
        mut,
        constraint = admin.key() == protocol_state.admin @ BrahmanError::Unauthorized
    )]
    pub admin: Signer<'info>,

    #[account(
        mut,
        seeds = [b"brahman-protocol-state"],
        bump,
    )]
    pub protocol_state: Account<'info, ProtocolState>,
}

#[derive(Accounts)]
pub struct FinalizeVerification<'info> {
    #[account(
        mut,
        constraint = authority.key() == protocol_state.admin @ BrahmanError::UnauthorizedFinalization
    )]
    pub authority: Signer<'info>,

    #[account(mut)]
    pub verification_record: Account<'info, VerificationRecord>,

    #[account(
        mut,
        seeds = [b"brahman-protocol-state"],
        bump,
    )]
    pub protocol_state: Account<'info, ProtocolState>,
}

// ─────────────────────────────────────────────────────────────
// Account State
// ─────────────────────────────────────────────────────────────

#[account]
pub struct ProtocolState {
    pub admin: Pubkey,
    pub min_quorum: u8,
    pub quorum_threshold_bps: u16,
    pub total_verifications: u64,
    pub total_verdicts_valid: u64,
    pub total_verdicts_invalid: u64,
    pub total_verdicts_disputed: u64,
    pub registered_node_count: u8,
    pub registered_nodes: [Pubkey; 32],
}

impl Default for ProtocolState {
    fn default() -> Self {
        Self {
            admin: Pubkey::default(),
            min_quorum: 0,
            quorum_threshold_bps: 0,
            total_verifications: 0,
            total_verdicts_valid: 0,
            total_verdicts_invalid: 0,
            total_verdicts_disputed: 0,
            registered_node_count: 0,
            registered_nodes: [Pubkey::default(); 32],
        }
    }
}

impl ProtocolState {
    // 8 (discriminator) + 32 + 1 + 2 + 8 + 8 + 8 + 8 + 1 + (32*32) = 1100
    pub const SPACE: usize = 8 + 32 + 1 + 2 + 8 + 8 + 8 + 8 + 1 + (32 * 32);
}

#[account]
pub struct VerificationRecord {
    pub logic_hash: [u8; 32],
    pub is_verified: bool,
    pub quorum_status: u8,
    pub final_verdict: u8,
    pub final_logic_hash: [u8; 32],
    pub created_at: i64,
    pub finalized_at: i64,
    pub authority: Pubkey,
    pub vote_count: u8,
    pub voter_pubkeys: [Pubkey; 10],
    pub voter_verdicts: [u8; 10],
    pub voter_hashes: [[u8; 32]; 10],
}

impl Default for VerificationRecord {
    fn default() -> Self {
        Self {
            logic_hash: [0u8; 32],
            is_verified: false,
            quorum_status: 0,
            final_verdict: 0,
            final_logic_hash: [0u8; 32],
            created_at: 0,
            finalized_at: 0,
            authority: Pubkey::default(),
            vote_count: 0,
            voter_pubkeys: [Pubkey::default(); 10],
            voter_verdicts: [0u8; 10],
            voter_hashes: [[0u8; 32]; 10],
        }
    }
}

impl VerificationRecord {
    // 8 + 32 + 1 + 1 + 1 + 32 + 8 + 8 + 32 + 1 + 320 + 10 + 320 = 774
    pub const SPACE: usize = 8 + 32 + 1 + 1 + 1 + 32 + 8 + 8 + 32
        + 1 + 320 + 10 + 320;
}

// ─────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────

#[error_code]
pub enum BrahmanError {
    #[msg("Verification already finalized")]
    AlreadyFinalized,
    #[msg("Duplicate vote from same node")]
    DuplicateVote,
    #[msg("Maximum votes reached (10)")]
    MaxVotesReached,
    #[msg("Insufficient votes for quorum")]
    InsufficientVotes,
    #[msg("Unauthorized")]
    Unauthorized,
    #[msg("Unauthorized initialization — deployer mismatch")]
    UnauthorizedInitialization,
    #[msg("Unauthorized node — not registered in protocol state")]
    UnauthorizedNode,
    #[msg("Unauthorized finalization — admin only")]
    UnauthorizedFinalization,
    #[msg("Maximum registered nodes reached (32)")]
    MaxNodesReached,
    #[msg("Node already registered")]
    DuplicateNode,
}
