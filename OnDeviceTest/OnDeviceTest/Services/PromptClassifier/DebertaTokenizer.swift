/*
 DebertaTokenizer.swift
 
 Custom SentencePiece-style tokenizer for DeBERTa models.
 
 Why Custom Implementation?
 The swift-transformers library doesn't support DeBERTaV2Tokenizer,
 so we implement a simplified version that's compatible with the
 NVIDIA prompt classifier model.
 
 Algorithm: Greedy Longest-Match
 1. Start with the full text
 2. Find the longest token in vocab that matches the start
 3. Add that token, advance position
 4. Repeat until end of text
 
 Special Tokens:
 - [CLS]: Start of sequence (required by BERT-style models)
 - [SEP]: End of sequence / separator
 - [PAD]: Padding for fixed-length sequences
 - [UNK]: Unknown tokens not in vocabulary
 
 Required Files (in app bundle):
 - vocab.json: Token -> ID mapping (~128k entries)
 
 Note: This is a simplified tokenizer. For production use,
 consider using the full SentencePiece library via SPM.
 */

import Foundation

/// A simple SentencePiece-style tokenizer for DeBERTa
/// Uses greedy longest-match tokenization with the vocab
class DebertaTokenizer {
    private var vocab: [String: Int] = [:]
    private var reverseVocab: [Int: String] = [:]
    
    // Special tokens
    let clsToken = "[CLS]"
    let sepToken = "[SEP]"
    let padToken = "[PAD]"
    let unkToken = "[UNK]"
    
    var clsId: Int { vocab[clsToken] ?? 1 }
    var sepId: Int { vocab[sepToken] ?? 2 }
    var padId: Int { vocab[padToken] ?? 0 }
    var unkId: Int { vocab[unkToken] ?? 3 }
    
    init() throws {
        // Load vocab from vocab.json in bundle
        guard let vocabUrl = Bundle.main.url(forResource: "vocab", withExtension: "json") else {
            throw NSError(domain: "DebertaTokenizer", code: 1, userInfo: [NSLocalizedDescriptionKey: "vocab.json not found in bundle"])
        }
        
        let data = try Data(contentsOf: vocabUrl)
        guard let loadedVocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw NSError(domain: "DebertaTokenizer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to parse vocab.json"])
        }
        
        self.vocab = loadedVocab
        
        // Build reverse vocab
        for (token, id) in vocab {
            reverseVocab[id] = token
        }
        
        print("DebertaTokenizer loaded with \(vocab.count) tokens")
    }
    
    /// Encode text to token IDs
    func encode(text: String) -> [Int] {
        var ids: [Int] = [clsId]
        
        // Normalize: lowercase and add space prefix (SentencePiece style)
        let normalizedText = " " + text.lowercased()
        
        // Tokenize using greedy longest-match
        let tokens = tokenize(normalizedText)
        
        for token in tokens {
            if let id = vocab[token] {
                ids.append(id)
            } else if let id = vocab["▁" + token] {
                // Try with SentencePiece prefix
                ids.append(id)
            } else {
                // Unknown token - try character by character
                for char in token {
                    let charStr = String(char)
                    if let id = vocab[charStr] {
                        ids.append(id)
                    } else if let id = vocab["▁" + charStr] {
                        ids.append(id)
                    } else {
                        ids.append(unkId)
                    }
                }
            }
        }
        
        ids.append(sepId)
        return ids
    }
    
    /// Greedy longest-match tokenization
    private func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var remaining = text
        
        while !remaining.isEmpty {
            var found = false
            
            // Try to find longest matching token
            for length in stride(from: min(remaining.count, 20), through: 1, by: -1) {
                let endIndex = remaining.index(remaining.startIndex, offsetBy: length)
                let candidate = String(remaining[..<endIndex])
                
                // Check if this token exists in vocab (with or without ▁ prefix)
                if vocab[candidate] != nil || vocab["▁" + candidate] != nil {
                    tokens.append(candidate)
                    remaining = String(remaining[endIndex...])
                    found = true
                    break
                }
            }
            
            // If no match found, take single character
            if !found {
                let char = String(remaining.removeFirst())
                if !char.trimmingCharacters(in: .whitespaces).isEmpty {
                    tokens.append(char)
                }
            }
        }
        
        return tokens
    }
    
    /// Decode token IDs back to text
    func decode(ids: [Int]) -> String {
        var text = ""
        for id in ids {
            if let token = reverseVocab[id] {
                // Skip special tokens
                if token == clsToken || token == sepToken || token == padToken {
                    continue
                }
                // Remove SentencePiece prefix and join
                let cleaned = token.replacingOccurrences(of: "▁", with: " ")
                text += cleaned
            }
        }
        return text.trimmingCharacters(in: .whitespaces)
    }
}

