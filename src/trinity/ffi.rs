// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::sync::Mutex;

use super::Trinity;
use super::cgt_bridge::{CGTConjecture, ConjectureInjector, ConjectureValidator};

static mut TRINITY_INSTANCE: Option<Mutex<Trinity>> = None;

#[no_mangle]
pub extern "C" fn yp_trinity_init() -> *mut c_void {
    let trinity = Trinity::default();
    unsafe {
        TRINITY_INSTANCE = Some(Mutex::new(trinity));
    }
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn yp_trinity_wallet_status() -> *mut c_char {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            let mut status = String::from("{");
            for (domain, wallet) in &trinity.predictor.wallets {
                let domain_str = match domain {
                    super::predictor::Domain::CS => "CS",
                    super::predictor::Domain::Medical => "Medical",
                    super::predictor::Domain::Legal => "Legal",
                    super::predictor::Domain::General => "General",
                };
                status.push_str(&format!("\"{}\":{},", domain_str, wallet.balance));
            }
            status.pop();
            status.push('}');
            CString::new(status).unwrap().into_raw()
        } else {
            CString::new("{\"error\":\"not_initialized\"}").unwrap().into_raw()
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_is_canonicalized(pattern_json: *const c_char) -> c_int {
    if pattern_json.is_null() { return 0; }
    let json_str = unsafe { CStr::from_ptr(pattern_json).to_string_lossy() };
    
    let pattern: super::canonicalization::QueryPattern = match serde_json::from_str(&json_str) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            if trinity.canonicalization.is_canonicalized(&pattern) {
                return 1;
            }
        }
        0
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_record_query(query_json: *const c_char) -> c_int {
    if query_json.is_null() { return -1; }
    let json_str = unsafe { CStr::from_ptr(query_json).to_string_lossy() };
    
    #[derive(serde::Deserialize)]
    struct QueryRecord {
        query_id: u64,
        hash_prefix: [u8; 4],
        domain: String,
        predicted_bucket: u32,
        predicted_engine: String,
        actual_bucket: u32,
        actual_engine: String,
        latency_ns: u64,
    }
    
    let record: QueryRecord = match serde_json::from_str(&json_str) {
        Ok(r) => r,
        Err(_) => return -2,
    };
    
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let mut trinity = mutex.lock().unwrap();
            
            // Record in provenance
            let trace = super::provenance::ExecutionTrace {
                query_id: record.query_id,
                predictor_hint: Some(format!("{}:{}", record.predicted_engine, record.predicted_bucket)),
                executor_action: format!("{}:{}", record.actual_engine, record.actual_bucket),
                execution_mode: if record.predicted_bucket == record.actual_bucket {
                    super::provenance::ExecutionMode::FollowedHint
                } else {
                    super::provenance::ExecutionMode::IgnoredHint
                },
                proof_hash: [0u8; 32],
            };
            trinity.provenance.append(trace);
            
            // Record in auditor ledger
            let hint = super::predictor::PredictorHint {
                query_id: record.query_id,
                predicted_bucket: record.predicted_bucket,
                predicted_engine: record.predicted_engine,
                predicted_latency_ns: record.latency_ns,
                confidence: 0.9,
                prewarm_buckets: vec![],
                domain: super::predictor::Domain::CS,
                source: "ffi_record".to_string(),
                tangent_id: None,
            };
            trinity.auditor.record(
                record.query_id,
                Some(hint),
                record.actual_bucket,
                record.actual_engine,
                super::provenance::ExecutionTrace {
                    query_id: record.query_id,
                    predictor_hint: None,
                    executor_action: "recorded".to_string(),
                    execution_mode: super::provenance::ExecutionMode::FollowedHint,
                    proof_hash: [0u8; 32],
                },
                {
                    let mut full_prefix = [0u8; 8];
                    full_prefix[..4].copy_from_slice(&record.hash_prefix);
                    full_prefix
                },
            );
            
            // Update canonicalization
            let pattern = super::canonicalization::QueryPattern {
                hash_prefix: record.hash_prefix,
                domain: record.domain,
            };
            let correct = record.predicted_bucket == record.actual_bucket;
            trinity.canonicalization.record(&pattern, correct);
            
            1
        } else {
            -3
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_inject_conjecture(json_ptr: *const c_char) -> c_int {
    if json_ptr.is_null() { return -1; }
    let json_str = unsafe { CStr::from_ptr(json_ptr).to_string_lossy() };
    
    let conjecture: CGTConjecture = match serde_json::from_str(&json_str) {
        Ok(c) => c,
        Err(_) => return -2,
    };
    
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let mut trinity = mutex.lock().unwrap();
            match ConjectureInjector::inject(&mut trinity.predictor, conjecture) {
                Ok(id) => id as c_int,
                Err(_) => -3,
            }
        } else {
            -4
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_conjecture_report() -> *mut c_char {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            let reports = ConjectureValidator::report(&trinity.predictor);
            let json = serde_json::to_string(&reports).unwrap_or_else(|_| "[]".to_string());
            CString::new(json).unwrap().into_raw()
        } else {
            CString::new("[]").unwrap().into_raw()
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_canonicalized_count() -> u64 {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            trinity.canonicalization.canonicalized_count() as u64
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_provenance_chain_len() -> u64 {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            trinity.provenance.len() as u64
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_free_string(s: *mut c_char) {
    unsafe {
        if !s.is_null() {
            let _ = CString::from_raw(s);
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_predict_bucket(query_id: u64) -> c_int {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            if let Some(hint) = trinity.predictor.predict(query_id) {
                return hint.predicted_bucket as c_int;
            }
        }
        -1
    }
}

#[no_mangle]
pub extern "C" fn yp_trinity_record_result(query_id: u64, predicted: c_int, actual: c_int) {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let mut trinity = mutex.lock().unwrap();
            // Memorize the actual bucket for this query so future predictions can hit.
            trinity.predictor.record(query_id, actual as u32);
            let change = if predicted == actual { 1 } else { -10 };
            let reason = if predicted == actual { "hint_correct" } else { "hint_wrong" };
            trinity.predictor.update_wallet(
                &super::predictor::Domain::CS,
                change,
                reason,
                query_id as u64,
            );
        }
    }
}

/// Export Trinity predictor trace as JSON string for CGT rerank.
/// Caller must free the returned pointer with yp_trinity_free_string.
#[no_mangle]
pub extern "C" fn yp_trinity_predictor_trace(query_id: u64) -> *mut c_char {
    unsafe {
        if let Some(ref mutex) = TRINITY_INSTANCE {
            let trinity = mutex.lock().unwrap();
            let trace = trinity.predictor.predictor_trace(query_id);
            let json = serde_json::to_string(&trace).unwrap_or_else(|_| "{}".to_string());
            CString::new(json).unwrap().into_raw()
        } else {
            CString::new("{}").unwrap().into_raw()
        }
    }
}
