use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::hash_map::HashMap;

const BASE_VOCAB: usize = 256;
const TERMINATOR: u64 = 256;
const CHUNK: usize = 512;

type Token = u64;
type TokenPair = (Token, Token);
type PairDistribution = HashMap<TokenPair, u64>;

struct CodecCore {
    pair_to_token: HashMap<TokenPair, Token>,
}

#[pyclass(name = "_BpeCodec", frozen)]
struct BpeCodec {
    core: CodecCore,
    token_objects: Vec<Py<PyInt>>,
}

impl CodecCore {
    fn from_pairs(pairs: Vec<TokenPair>) -> Result<Self, String> {
        let mut pair_to_token = HashMap::with_capacity(pairs.len());

        for (rank, &(left, right)) in pairs.iter().enumerate() {
            let token = (BASE_VOCAB + 1 + rank) as Token;
            if left == TERMINATOR || right == TERMINATOR {
                return Err(format!("merge {token} references the terminator"));
            }
            if left >= token {
                return Err(format!("merge {token} references unknown token {left}"));
            }
            if right >= token {
                return Err(format!("merge {token} references unknown token {right}"));
            }

            pair_to_token.insert((left, right), token);
        }

        Ok(Self { pair_to_token })
    }

    fn encode_bytes(&self, bytes: &[u8]) -> Vec<Token> {
        let mut tokens: Vec<Token> = bytes.iter().map(|&byte| byte.into()).collect();
        if tokens.len() < 2 {
            tokens.push(TERMINATOR);
            return tokens;
        }

        // Nodes retain their original positions. Merging detaches the right-hand
        // node with two link updates; obsolete heap entries are discarded lazily.
        let mut previous: Vec<Option<usize>> = (0..tokens.len())
            .map(|index| index.checked_sub(1))
            .collect();
        let mut following: Vec<Option<usize>> = (0..tokens.len())
            .map(|index| (index + 1 < tokens.len()).then_some(index + 1))
            .collect();

        // Token ids increase with merge rank, so Reverse gives us the same
        // (rank, left position, right position) priority as Python's min-heap.
        let mut candidates = BinaryHeap::new();
        for left in 0..tokens.len() - 1 {
            if let Some(&merged) = self.pair_to_token.get(&(tokens[left], tokens[left + 1])) {
                candidates.push(Reverse((merged, left, left + 1)));
            }
        }

        while let Some(Reverse((merged, left, right))) = candidates.pop() {
            if following[left] != Some(right) {
                continue;
            }
            if self.pair_to_token.get(&(tokens[left], tokens[right])) != Some(&merged) {
                continue;
            }

            let before = previous[left];
            let after = following[right];
            tokens[left] = merged;
            following[left] = after;
            if let Some(after) = after {
                previous[after] = Some(left);
            }

            previous[right] = None;
            following[right] = None;

            if let Some(before) = before {
                if let Some(&before_merged) =
                    self.pair_to_token.get(&(tokens[before], tokens[left]))
                {
                    candidates.push(Reverse((before_merged, before, left)));
                }
            }
            if let Some(after) = after {
                if let Some(&after_merged) = self.pair_to_token.get(&(tokens[left], tokens[after]))
                {
                    candidates.push(Reverse((after_merged, left, after)));
                }
            }
        }

        let mut encoded = Vec::with_capacity(tokens.len());
        let mut node = Some(0);
        while let Some(index) = node {
            encoded.push(tokens[index]);
            node = following[index];
        }
        encoded.push(TERMINATOR);
        encoded
    }
}

impl BpeCodec {
    fn tokens_to_list<'py>(
        &self,
        py: Python<'py>,
        tokens: &[Token],
    ) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            tokens
                .iter()
                .map(|&token| self.token_objects[token as usize].bind(py)),
        )
    }
}

#[pymethods]
impl BpeCodec {
    #[new]
    fn new(py: Python<'_>, pairs: Vec<TokenPair>) -> PyResult<Self> {
        let token_count = BASE_VOCAB + 1 + pairs.len();
        let core = CodecCore::from_pairs(pairs).map_err(PyValueError::new_err)?;
        let token_objects = (0..token_count)
            .map(|token| PyInt::new(py, token).unbind())
            .collect();
        Ok(Self {
            core,
            token_objects,
        })
    }

    fn encode<'py>(&self, py: Python<'py>, docs: Vec<String>) -> PyResult<Bound<'py, PyList>> {
        let encoded: Vec<Vec<Token>> = py.detach(move || {
            docs.par_iter()
                .map(|doc| self.core.encode_bytes(doc.as_bytes()))
                .collect()
        });
        let token_count = encoded.iter().map(Vec::len).sum();
        let mut flattened = Vec::with_capacity(token_count);
        for tokens in encoded {
            flattened.extend(tokens);
        }
        self.tokens_to_list(py, &flattened)
    }
}

#[pyfunction(name = "_train")]
fn train_native(_py: Python<'_>, docs: Vec<String>, vocab: usize) -> PyResult<Vec<(u64, u64)>> {
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
        .with_prefix("bpe.dist");
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
        .reduce(PairDistribution::new, |mut a, b| {
            for (k, v) in b {
                a.entry(k).and_modify(|c| *c += v).or_insert(v);
            }
            a
        });

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
        let tokens = vec![a, b];
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
                        let (na, nb) = pairs[tok as usize - BASE_VOCAB - 1];
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
    m.add_class::<BpeCodec>()?;
    m.add_function(wrap_pyfunction!(train_native, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_uses_merge_rank_and_leftmost_tie_breaking() {
        // Both initial pairs are mergeable, but (b, c) has the better rank.
        let codec = CodecCore::from_pairs(vec![
            (b'b'.into(), b'c'.into()),
            (b'a'.into(), b'b'.into()),
            (b'a'.into(), 257),
        ])
        .unwrap();
        assert_eq!(codec.encode_bytes(b"abc"), vec![259, TERMINATOR]);

        let codec =
            CodecCore::from_pairs(vec![(b'a'.into(), b'b'.into()), (257, b'c'.into())]).unwrap();
        assert_eq!(codec.encode_bytes(b"abc"), vec![258, TERMINATOR]);

        let codec = CodecCore::from_pairs(vec![(b'a'.into(), b'a'.into())]).unwrap();
        assert_eq!(
            codec.encode_bytes(b"aaa"),
            vec![257, b'a'.into(), TERMINATOR]
        );
    }

    #[test]
    fn encoding_handles_utf8_and_empty_documents() {
        let codec = CodecCore::from_pairs(Vec::new()).unwrap();
        let text = "naïve 🦀";
        let mut expected: Vec<Token> = text.bytes().map(Token::from).collect();
        expected.push(TERMINATOR);

        assert_eq!(codec.encode_bytes(text.as_bytes()), expected);
        assert_eq!(codec.encode_bytes(b""), vec![TERMINATOR]);
    }
}
