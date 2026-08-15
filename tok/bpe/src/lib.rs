// use bitset::BitSet;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::hash_map::HashMap;

const BASE_VOCAB: usize = 256;
const TERMINATOR: u64 = 256;
const CHUNK: usize = 512;

type Token = u64;
type TokenPair = (Token, Token);
pub type PairDistribution = HashMap<TokenPair, u64>;

#[pyfunction]
fn train_native(py: Python<'_>, docs: Vec<String>, vocab: usize) -> PyResult<Vec<(u64, u64)>> {
    println!("bpe.native.init");
    // let mut universe = Universe::new(docs).unwrap();
    let mut tokenized_docs: Vec<Vec<Token>> = docs
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.bytes().map(|b| b.into()).collect())
        .collect();
    let dist_bar = ProgressBar::new(tokenized_docs.len() as u64)
        .with_style(
            ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}]")
                .unwrap()
                .progress_chars("=> "),
        )
        .with_prefix(format!["bpe.dist"]);
    let mut dist = tokenized_docs
        .par_chunks(CHUNK)
        .map(|docs| {
            let mut dist = PairDistribution::new();
            for doc in docs {
                for i in 0..doc.len() - 1 {
                    let a = doc[i];
                    let b = doc[i + 1];
                    if a == TERMINATOR || b == TERMINATOR {
                        continue;
                    }
                    dist.entry((a, b)).and_modify(|i| *i += 1).or_insert(1);
                }
            }
            dist_bar.inc(CHUNK as u64);
            dist
        })
        .reduce(
            || PairDistribution::new(),
            |mut a, b| {
                for (k, v) in b {
                    a.entry(k).and_modify(|c| *c += v).or_insert(v);
                }
                a
            },
        );

    dist_bar.finish();

    let mut pairs = Vec::new();
    while BASE_VOCAB + pairs.len() < vocab - 1 {
        let max = dist
            .iter()
            .filter(|((a, b), c)| *a != TERMINATOR && *b != TERMINATOR && **c > 0)
            .max_by(|(_, v1), (_, v2)| v1.cmp(v2));
        let ((a, b), c) = match max {
            Some(((a, b), c)) => ((*a, *b), *c),
            None => break,
        };
        pairs.push((a, b));
        let new_token = (BASE_VOCAB + pairs.len()) as Token;
        let mut tokens = vec![a, b];
        let mut utf: Vec<u8> = vec![];
        for tok in tokens {
            if tok == TERMINATOR {
                break;
            }
            let mut expanded = vec![tok];
            let mut dirty = true;
            while dirty {
                dirty = false;
                let mut new_expanded = vec![];
                for tok in expanded {
                    if tok > BASE_VOCAB as Token {
                        let (na, nb) = pairs[tok as usize - BASE_VOCAB as usize - 1 as usize];
                        new_expanded.push(na);
                        new_expanded.push(nb);
                        dirty = true;
                    } else {
                        new_expanded.push(tok)
                    }
                }
                expanded = new_expanded
            }
            for exp in expanded {
                utf.push(exp as u8);
            }
        }
        let pair_string = String::from_utf8_lossy(&utf).clone();
        let merge_bar = ProgressBar::new(tokenized_docs.len() as u64)
            .with_style(
                ProgressStyle::with_template(
                    "{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] {msg}",
                )
                .unwrap()
                .progress_chars("=> "),
            )
            .with_prefix("bpe.merge")
            .with_message(format![
                "merging {} \"{}\"s ({}, {}) -> {}",
                c, pair_string, a, b, new_token
            ]);
        let (plus, minus) = tokenized_docs
            .par_chunks_mut(CHUNK)
            .map(|docs| {
                let mut minus = PairDistribution::new();
                let mut plus = PairDistribution::new();
                for doc in docs {
                    let mut i = 0;
                    let mut new_doc = vec![];
                    while i < doc.len() {
                        let ta = doc[i];
                        if i + 1 < doc.len() {
                            let tb = doc[i + 1];
                            if a == ta && b == tb {
                                // need to update distributon.
                                //
                                // init: a, b, c, d
                                // indices: i - 1, i, i + 1, i +2
                                // identifiers: lt, ta, tb, nt
                                // merge: b, c -> e
                                // new: a, e, d

                                // anything operating on existing tokens is tallied
                                // and thus we can do blind access

                                // actions:
                                // - decrement a, b -> lt, ta
                                // - increment a, e -> lt, new_token
                                if i > 0 {
                                    let lt = *new_doc.last().unwrap();
                                    minus.entry((lt, ta)).and_modify(|c| *c += 1).or_insert(1);
                                    plus.entry((lt, new_token))
                                        .and_modify(|c| *c += 1)
                                        .or_insert(1);
                                }

                                // - decrement b, c -> ta, tb
                                minus.entry((ta, tb)).and_modify(|c| *c += 1).or_insert(1);

                                // - decrement c, d -> tb, nt
                                // - increment e, d -> new_token, nnt
                                if i + 2 < doc.len() {
                                    let nt = doc[i + 2];
                                    minus.entry((tb, nt)).and_modify(|c| *c += 1).or_insert(1);
                                    plus.entry((new_token, nt))
                                        .and_modify(|c| *c += 1)
                                        .or_insert(1);
                                }
                                new_doc.push(new_token);
                                i += 2;
                            } else {
                                new_doc.push(ta);
                                i += 1;
                            }
                        } else {
                            new_doc.push(ta);
                            i += 1;
                        }
                    }
                    *doc = new_doc;
                }
                merge_bar.inc(CHUNK as u64);
                (plus, minus)
            })
            .reduce(
                || (PairDistribution::new(), PairDistribution::new()),
                |(mut plus_lhs, mut minus_lhs), (plus_rhs, minus_rhs)| {
                    for (k, v) in plus_rhs {
                        plus_lhs.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }

                    for (k, v) in minus_rhs {
                        minus_lhs.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }

                    (plus_lhs, minus_lhs)
                },
            );

        for (pair, delta) in plus {
            dist.entry(pair)
                .and_modify(|c| *c += delta)
                .or_insert(delta);
        }

        for (pair, delta) in minus {
            *dist.get_mut(&pair).unwrap() -= delta;
        }

        merge_bar.finish();
    }

    Ok(pairs)
}

#[pymodule]
fn bpe_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_native, m)?)?;
    Ok(())
}
