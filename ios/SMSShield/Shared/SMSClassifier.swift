import Foundation

/// Lightweight TF-IDF + Logistic Regression classifier for SMS messages.
/// Loads vocabulary, IDF weights, and logistic regression coefficients from a JSON file.
/// Produces identical predictions to the Python simplified model.
struct SMSClassifier {

    struct Prediction {
        let label: String            // "ham", "spam", or "smishing"
        let confidence: Double       // probability of predicted class
        let probabilities: [String: Double]  // all class probabilities
    }

    private let vocabulary: [String: Int]   // token -> feature index
    private let idf: [Double]               // inverse document frequency weights
    private let coefficients: [[Double]]    // (numClasses x numFeatures)
    private let intercept: [Double]         // (numClasses,)
    private let classes: [String]           // class labels in model order

    // MARK: - Initialization

    init(modelURL: URL) throws {
        let data = try Data(contentsOf: modelURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        guard let vocab = json["vocabulary"] as? [String: Int],
              let idfArr = json["idf"] as? [Double],
              let coefArr = json["coefficients"] as? [[Double]],
              let interceptArr = json["intercept"] as? [Double],
              let info = json["model_info"] as? [String: Any],
              let classArr = info["classes"] as? [String]
        else {
            throw NSError(domain: "SMSClassifier", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid model JSON format"])
        }

        self.vocabulary = vocab
        self.idf = idfArr
        self.coefficients = coefArr
        self.intercept = interceptArr
        self.classes = classArr
    }

    // MARK: - Prediction

    func predict(_ text: String) -> Prediction {
        let tfidfVector = computeTFIDF(text)
        let logits = computeLogits(tfidfVector)
        let probabilities = softmax(logits)

        var bestIdx = 0
        for i in 1..<probabilities.count {
            if probabilities[i] > probabilities[bestIdx] {
                bestIdx = i
            }
        }

        var probDict: [String: Double] = [:]
        for (i, cls) in classes.enumerated() {
            probDict[cls] = probabilities[i]
        }

        return Prediction(
            label: classes[bestIdx],
            confidence: probabilities[bestIdx],
            probabilities: probDict
        )
    }

    // MARK: - TF-IDF Computation

    private func computeTFIDF(_ text: String) -> [Double] {
        let tokens = tokenize(text.lowercased())
        let numFeatures = idf.count

        // Count term frequencies
        var tf = [Int: Int]()  // feature index -> count
        for token in tokens {
            if let idx = vocabulary[token] {
                tf[idx, default: 0] += 1
            }
        }

        // Also count bigrams
        for i in 0..<(tokens.count - 1) {
            let bigram = "\(tokens[i]) \(tokens[i + 1])"
            if let idx = vocabulary[bigram] {
                tf[idx, default: 0] += 1
            }
        }

        // Compute TF-IDF with sublinear TF: 1 + log(tf)
        var vector = [Double](repeating: 0.0, count: numFeatures)
        for (idx, count) in tf {
            let sublinearTF = 1.0 + log(Double(count))
            vector[idx] = sublinearTF * idf[idx]
        }

        // L2 normalize
        let norm = sqrt(vector.reduce(0.0) { $0 + $1 * $1 })
        if norm > 0 {
            for i in 0..<vector.count {
                vector[i] /= norm
            }
        }

        return vector
    }

    private func tokenize(_ text: String) -> [String] {
        // Match sklearn's default token pattern: sequences of 2+ alphanumeric chars
        let pattern = #"(?u)\b\w\w+\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: range)
        return matches.compactMap { match in
            Range(match.range, in: text).map { String(text[$0]) }
        }
    }

    // MARK: - Logistic Regression

    private func computeLogits(_ features: [Double]) -> [Double] {
        var logits = intercept
        for c in 0..<coefficients.count {
            for (i, val) in features.enumerated() where val != 0 {
                logits[c] += coefficients[c][i] * val
            }
        }
        return logits
    }

    private func softmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}
