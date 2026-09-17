// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! CGT Model Verifier — checks the 14 defining properties
//! against the formal axioms in cgt/docs/model_spec.md

/// 8x64 bit matrix as row-major Vec<u8>
pub struct Matrix8x64 {
    pub data: [u8; 64], // 8 rows x 64 bits, packed as bytes
}

impl Matrix8x64 {
    pub fn get(&self, row: usize, col: usize) -> u8 {
        let byte_idx = row * 8 + (col / 8);
        let bit_idx = col % 8;
        (self.data[byte_idx] >> bit_idx) & 1
    }

    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        let byte_idx = row * 8 + (col / 8);
        let bit_idx = col % 8;
        if val == 1 {
            self.data[byte_idx] |= 1 << bit_idx;
        } else {
            self.data[byte_idx] &= !(1 << bit_idx);
        }
    }

    pub fn bits(&self) -> u32 {
        self.data.iter().map(|b| b.count_ones()).sum()
    }

    pub fn h(&self) -> u32 {
        let mut sum = 0u32;
        for r in 0..8 {
            for c in 0..63 {
                sum += (self.get(r, c) & self.get(r, c + 1)) as u32;
            }
        }
        sum
    }

    pub fn v(&self) -> u32 {
        let mut sum = 0u32;
        for c in 0..64 {
            for r in 0..7 {
                sum += (self.get(r, c) & self.get(r + 1, c)) as u32;
            }
        }
        sum
    }

    pub fn x2(&self) -> u32 {
        let mut sum = 0u32;
        for r in 0..7 {
            for c in 0..63 {
                sum += (self.get(r, c) & self.get(r, c + 1) & self.get(r + 1, c) & self.get(r + 1, c + 1)) as u32;
            }
        }
        sum
    }

    pub fn chi(&self) -> i32 {
        self.bits() as i32 - self.h() as i32 - self.v() as i32 + 2 * self.x2() as i32
    }
}

/// Build a matrix with ones at specific queue positions
pub fn matrix_with_queues(positions: &[(usize, usize)]) -> Matrix8x64 {
    let mut m = Matrix8x64 { data: [0u8; 64] };
    for &(r, c) in positions {
        m.set(r, c, 1);
    }
    m
}

/// Verify all 14 defining properties
pub fn verify_all_properties() -> Vec<(String, bool, i32, i32)> {
    let mut results = Vec::with_capacity(14);

    // Helper: expected chi for n isolated 1-bits
    let isolated_chi = |n: i32| n;

    // Helper: expected chi for n adjacent pairs (horizontal or vertical)
    let adjacent_chi = |n: i32| n; // bits=n*2, h or v=n, x2=0 => chi = 2n - n - 0 + 0 = n

    // 1. Q00, Q01 — adjacent horizontal
    let m = matrix_with_queues(&[(0, 0), (0, 1)]);
    let expected = 2 - 1 - 0 + 0; // bits=2, h=1, v=0, x2=0
    results.push(("Q00_Q01_adjacent_h".to_string(), m.chi() == expected, m.chi(), expected));

    // 2. Q00, Q10 — adjacent vertical
    let m = matrix_with_queues(&[(0, 0), (1, 0)]);
    let expected = 2 - 0 - 1 + 0;
    results.push(("Q00_Q10_adjacent_v".to_string(), m.chi() == expected, m.chi(), expected));

    // 3. Q00, Q11 — diagonal
    let m = matrix_with_queues(&[(0, 0), (1, 1)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q00_Q11_diagonal".to_string(), m.chi() == expected, m.chi(), expected));

    // 4. Q00, Q22 — separated (2,2)
    let m = matrix_with_queues(&[(0, 0), (2, 2)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q00_Q22_separated".to_string(), m.chi() == expected, m.chi(), expected));

    // 5. Q00, Q02 — same row, gap 1
    let m = matrix_with_queues(&[(0, 0), (0, 2)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q00_Q02_gap1_h".to_string(), m.chi() == expected, m.chi(), expected));

    // 6. Q00, Q20 — same col, gap 1
    let m = matrix_with_queues(&[(0, 0), (2, 0)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q00_Q20_gap1_v".to_string(), m.chi() == expected, m.chi(), expected));

    // 7. Q01, Q11 — L-shape
    let m = matrix_with_queues(&[(0, 1), (1, 1)]);
    let expected = 2 - 0 - 1 + 0;
    results.push(("Q01_Q11_lshape".to_string(), m.chi() == expected, m.chi(), expected));

    // 8. Q10, Q11 — L-shape rotated
    let m = matrix_with_queues(&[(1, 0), (1, 1)]);
    let expected = 2 - 1 - 0 + 0;
    results.push(("Q10_Q11_lshape_rot".to_string(), m.chi() == expected, m.chi(), expected));

    // 9. Q00, Q00 (self) — identity, but two separate 1-bits at same position is just one bit
    // Instead: single 1-bit
    let m = matrix_with_queues(&[(0, 0)]);
    let expected = 1 - 0 - 0 + 0;
    results.push(("Q00_single".to_string(), m.chi() == expected, m.chi(), expected));

    // 10. Q00, Q07 — full row apart
    let m = matrix_with_queues(&[(0, 0), (0, 7)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q00_Q07_far_h".to_string(), m.chi() == expected, m.chi(), expected));

    // 11. Q00, Q70 — full col apart
    let m = matrix_with_queues(&[(0, 0), (7, 0)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q00_Q70_far_v".to_string(), m.chi() == expected, m.chi(), expected));

    // 12. Q33, Q34 — adjacent center
    let m = matrix_with_queues(&[(3, 3), (3, 4)]);
    let expected = 2 - 1 - 0 + 0;
    results.push(("Q33_Q34_center_h".to_string(), m.chi() == expected, m.chi(), expected));

    // 13. Q33, Q43 — adjacent center (vert)
    let m = matrix_with_queues(&[(3, 3), (4, 3)]);
    let expected = 2 - 0 - 1 + 0;
    results.push(("Q33_Q43_center_v".to_string(), m.chi() == expected, m.chi(), expected));

    // 14. Q33, Q44 — diagonal center
    let m = matrix_with_queues(&[(3, 3), (4, 4)]);
    let expected = 2 - 0 - 0 + 0;
    results.push(("Q33_Q44_center_diag".to_string(), m.chi() == expected, m.chi(), expected));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_14_properties() {
        let results = verify_all_properties();
        let mut passed = 0;
        for (name, ok, actual, expected) in &results {
            if *ok {
                passed += 1;
                println!("  PASS {name}: chi={actual} (expected {expected})");
            } else {
                println!("  FAIL {name}: chi={actual} (expected {expected})");
            }
        }
        println!("CGT properties: {passed}/{} passed", results.len());
        assert_eq!(passed, results.len(), "Some CGT defining properties failed");
    }

    #[test]
    fn test_axiom1_formula() {
        let m = matrix_with_queues(&[(0,0), (0,1), (1,0), (1,1)]);
        // bits=4, h=2 (row0: 0,1; row1: 0,1), v=2 (col0: 0,1; col1: 0,1), x2=1
        // chi = 4 - 2 - 2 + 2*1 = 2
        assert_eq!(m.bits(), 4);
        assert_eq!(m.h(), 2);
        assert_eq!(m.v(), 2);
        assert_eq!(m.x2(), 1);
        assert_eq!(m.chi(), 2);
    }
}
