//
//  BabyNamesOnDeviceRegression.swift
//  AISampleApp
//
//  Loads babyNames.csv (State,Sex,Year,Name,Count), subsamples rows on-device,
//  trains Create ML MLLinearRegressor → Core ML MLModel, then runs sample predictions.
//

import CoreML
import CreateML
import Foundation

enum BabyNamesRegressionError: LocalizedError {
    case securityScopeDenied
    case emptySample
    case parseFailed(String)
    case trainingFailed(String)

    var errorDescription: String? {
        switch self {
        case .securityScopeDenied:
            return "Could not access the selected file."
        case .emptySample:
            return "No valid data rows were found in the CSV."
        case .parseFailed(let line):
            return "Could not parse a CSV line: \(line.prefix(80))…"
        case .trainingFailed(let message):
            return message
        }
    }
}

/// Parses `State,Sex,Year,Name,Count` allowing commas inside `Name`.
private func parseBabyNamesFullRow(_ line: String) -> (year: Int, sex: String, name: String, count: Double)? {
    let parts = line.split(separator: ",", omittingEmptySubsequences: false).map {
        String($0).trimmingCharacters(in: .whitespaces)
    }
    guard parts.count >= 5 else { return nil }
    guard let year = Int(parts[2]) else { return nil }
    guard let count = Double(parts[parts.count - 1]) else { return nil }
    let sex = parts[1]
    guard sex == "M" || sex == "F" else { return nil }
    let name = parts[3..<(parts.count - 1)].joined(separator: ",").trimmingCharacters(in: .whitespaces)
    guard !name.isEmpty else { return nil }
    return (year, sex, name, count)
}

private func parseBabyNamesRow(_ line: String) -> (year: Int, sex: String, count: Double)? {
    guard let full = parseBabyNamesFullRow(line) else { return nil }
    return (full.year, full.sex, full.count)
}

/// Reservoir sample up to `maxRows` data rows (excluding header).
private func reservoirSampleRows(from url: URL, maxRows: Int) throws -> [(year: Int, sex: String, count: Double)] {
    guard url.startAccessingSecurityScopedResource() else {
        throw BabyNamesRegressionError.securityScopeDenied
    }
    defer { url.stopAccessingSecurityScopedResource() }

    let data = try Data(contentsOf: url)
    guard let text = String(data: data, encoding: .utf8) else {
        throw BabyNamesRegressionError.trainingFailed("The file is not valid UTF-8 text.")
    }

    var lines = text.split(whereSeparator: \.isNewline).map(String.init)
    if let first = lines.first,
       first.localizedCaseInsensitiveContains("year"),
       first.localizedCaseInsensitiveContains("count") {
        lines.removeFirst()
    }

    var reservoir: [(year: Int, sex: String, count: Double)] = []
    reservoir.reserveCapacity(Swift.min(maxRows, 1024))
    var n = 0

    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        guard let row = parseBabyNamesRow(trimmed) else { continue }
        n += 1
        if reservoir.count < maxRows {
            reservoir.append(row)
        } else {
            let r = Int.random(in: 1 ... n)
            if r <= maxRows {
                reservoir[r - 1] = row
            }
        }
    }

    guard !reservoir.isEmpty else { throw BabyNamesRegressionError.emptySample }
    return reservoir
}

/// Trains a linear regression model (Create ML) and returns the Core ML `MLModel` plus evaluation text.
func trainBabyNamesCountRegressor(from csvURL: URL, sampleRowCap: Int = 40_000) throws -> (model: MLModel, summary: String, sampleComparisons: [String]) {
    let rows = try reservoirSampleRows(from: csvURL, maxRows: sampleRowCap)

    let years = rows.map(\.year)
    let sexes = rows.map(\.sex)
    let counts = rows.map(\.count)

    let table = try MLDataTable(dictionary: [
        "Year": years,
        "Sex": sexes,
        "Count": counts,
    ])

    let parameters = MLLinearRegressor.ModelParameters()
    let regressor = try MLLinearRegressor(
        trainingData: table,
        targetColumn: "Count",
        featureColumns: ["Year", "Sex"],
        parameters: parameters
    )

    let model = regressor.model
    let trainingMetrics = regressor.trainingMetrics
    let rmse = trainingMetrics.rootMeanSquaredError

    let eval = try regressor.evaluation(on: table)
    let evalRMSE = eval.rootMeanSquaredError

    var summary = "Trained on \(rows.count) rows (reservoir sample, cap \(sampleRowCap)).\n"
    summary += String(format: "Training RMSE: %.2f\n", rmse)
    summary += String(format: "Evaluation RMSE (same sample): %.2f\n", evalRMSE)

    // Core ML predictions on a few held rows from the sample (not true holdout; demo only).
    let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "Count"
    var comparisons: [String] = []
    let previewIndices = [0, rows.count / 4, rows.count / 2].filter { $0 < rows.count }
    for i in previewIndices {
        let row = rows[i]
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "Year": MLFeatureValue(int64: Int64(row.year)),
            "Sex": MLFeatureValue(string: row.sex),
        ])
        let prediction = try model.prediction(from: input)
        guard let predicted = prediction.featureValue(for: outputName)?.doubleValue else { continue }
        let line = String(
            format: "Year %d Sex %@ — actual count %.0f, predicted %.1f",
            row.year,
            row.sex,
            row.count,
            predicted
        )
        comparisons.append(line)
    }

    return (model, summary, comparisons)
}

/// Sums `Count` for each `Name` in the given calendar year (all states and sexes), then returns the top `limit` names by total count.
func topBabyNamesAggregated(forYear year: Int, csvURL: URL, limit: Int = 10) throws -> [(name: String, totalCount: Double)] {
    let (nameYear, _) = try buildNameYearCountIndex(from: csvURL)
    return exactTopNames(forYear: year, nameYear: nameYear, limit: limit)
}

// MARK: - Exact vs extrapolated top names

/// One full scan: `name -> (year -> summed count)` and the set of calendar years that appear in the file.
func buildNameYearCountIndex(from csvURL: URL) throws -> (nameYear: [String: [Int: Double]], yearsPresent: Set<Int>) {
    guard csvURL.startAccessingSecurityScopedResource() else {
        throw BabyNamesRegressionError.securityScopeDenied
    }
    defer { csvURL.stopAccessingSecurityScopedResource() }

    let data = try Data(contentsOf: csvURL)
    guard let text = String(data: data, encoding: .utf8) else {
        throw BabyNamesRegressionError.trainingFailed("The file is not valid UTF-8 text.")
    }

    var nameYear: [String: [Int: Double]] = [:]
    var yearsPresent: Set<Int> = []
    var lines = text.split(whereSeparator: \.isNewline).map(String.init)
    if let first = lines.first,
       first.localizedCaseInsensitiveContains("Year"),
       first.localizedCaseInsensitiveContains("Count") {
        lines.removeFirst()
    }

    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        guard let row = parseBabyNamesFullRow(trimmed) else { continue }
        yearsPresent.insert(row.year)
        var byYear = nameYear[row.name, default: [:]]
        byYear[row.year, default: 0] += row.count
        nameYear[row.name] = byYear
    }

    guard !yearsPresent.isEmpty else { throw BabyNamesRegressionError.emptySample }
    return (nameYear, yearsPresent)
}

private func exactTopNames(
    forYear year: Int,
    nameYear: [String: [Int: Double]],
    limit: Int
) -> [(name: String, totalCount: Double)] {
    var pairs: [(String, Double)] = []
    pairs.reserveCapacity(1024)
    for (name, byYear) in nameYear {
        if let c = byYear[year], c > 0 {
            pairs.append((name, c))
        }
    }
    return pairs.sorted { $0.1 > $1.1 }.prefix(limit).map { (name: $0.0, totalCount: $0.1) }
}

/// Cholesky A = L Lᵀ with L lower triangular; solves A x = b via forward and backward substitution.
private func choleskySolve(A: [[Double]], b: [Double]) -> [Double]? {
    let n = A.count
    guard n > 0, b.count == n, A.allSatisfy({ $0.count == n }) else { return nil }
    var L = Array(repeating: Array(repeating: 0.0, count: n), count: n)
    for i in 0..<n {
        for j in 0...i {
            var sum = A[i][j]
            for k in 0..<j {
                sum -= L[i][k] * L[j][k]
            }
            if i == j {
                guard sum > 0 else { return nil }
                L[i][j] = sqrt(sum)
            } else {
                L[i][j] = sum / L[j][j]
            }
        }
    }
    var y = Array(repeating: 0.0, count: n)
    for i in 0..<n {
        var sum = b[i]
        for k in 0..<i {
            sum -= L[i][k] * y[k]
        }
        y[i] = sum / L[i][i]
    }
    var x = Array(repeating: 0.0, count: n)
    for i in (0..<n).reversed() {
        var sum = y[i]
        for k in (i + 1)..<n {
            sum -= L[k][i] * x[k]
        }
        x[i] = sum / L[i][i]
    }
    return x
}

/// Accumulates XᵀX and Xᵀt for linear regression on `log1p(count)` with feature row `φ(year)`.
private func accumulateNormalEquations(
    points: [(year: Int, count: Double)],
    periodYears: Double,
    useCycle: Bool,
    ridge: Double
) -> (XtX: [[Double]], Xty: [Double])? {
    let p = useCycle ? 4 : 2
    guard points.count >= p else { return nil }
    var XtX = Array(repeating: Array(repeating: 0.0, count: p), count: p)
    var Xty = Array(repeating: 0.0, count: p)
    for i in 0..<p {
        XtX[i][i] += ridge
    }
    for pt in points {
        let y = Double(pt.year)
        let t = log1p(Swift.max(0, pt.count))
        let row: [Double]
        if useCycle {
            let th = 2 * Double.pi * y / periodYears
            row = [1, y, sin(th), cos(th)]
        } else {
            row = [1, y]
        }
        for i in 0..<p {
            Xty[i] += row[i] * t
            for j in 0..<p {
                XtX[i][j] += row[i] * row[j]
            }
        }
    }
    return (XtX, Xty)
}

private func featureVector(year: Int, periodYears: Double, useCycle: Bool) -> [Double] {
    let y = Double(year)
    if useCycle {
        let th = 2 * Double.pi * y / periodYears
        return [1, y, sin(th), cos(th)]
    }
    return [1, y]
}

private func dot(_ a: [Double], _ b: [Double]) -> Double {
    zip(a, b).map(*).reduce(0, +)
}

/// For each name, fits `log1p(count) ≈ w·φ(year)` with φ = [1, year, sin(2πy/P), cos(2πy/P)] when enough points, else `[1, year]`, and evaluates at `targetYear`.
private func extrapolatedTopNames(
    targetYear: Int,
    nameYear: [String: [Int: Double]],
    periodYears: Double,
    limit: Int
) -> [(name: String, predictedTotal: Double)] {
    var scored: [(String, Double)] = []
    scored.reserveCapacity(4096)

    for (name, byYear) in nameYear {
        let points: [(year: Int, count: Double)] = byYear.map { ($0.key, $0.value) }.sorted { $0.year < $1.year }
        let n = points.count
        guard n >= 2 else { continue }

        let useCycle = n >= 4
        guard let eq = accumulateNormalEquations(points: points, periodYears: periodYears, useCycle: useCycle, ridge: 1e-3),
              let w = choleskySolve(A: eq.XtX, b: eq.Xty)
        else { continue }

        let phi = featureVector(year: targetYear, periodYears: periodYears, useCycle: useCycle)
        guard w.count == phi.count else { continue }
        let z = dot(w, phi)
        guard z.isFinite else { continue }
        let pred = expm1(z)
        guard pred.isFinite, pred > 0 else { continue }
        scored.append((name, pred))
    }

    return scored.sorted { $0.1 > $1.1 }.prefix(limit).map { (name: $0.0, predictedTotal: $0.1) }
}

/// Top names for a calendar year: exact counts if that year appears in the CSV; otherwise pattern-based extrapolation from each name's time series (trend + long-period cycle).
func topBabyNamesExactOrExtrapolated(
    forYear year: Int,
    csvURL: URL,
    cyclePeriodYears: Double = 40,
    limit: Int = 10
) throws -> (names: [(name: String, value: Double)], source: BabyNamesYearResultSource, detail: String) {
    let (nameYear, yearsPresent) = try buildNameYearCountIndex(from: csvURL)

    if yearsPresent.contains(year) {
        let exact = exactTopNames(forYear: year, nameYear: nameYear, limit: limit)
        let names = exact.map { (name: $0.name, value: $0.totalCount) }
        let detail =
            "This year appears in the file; values are summed counts (all states and sexes) for that year."
        return (names, .exactFromCSV, detail)
    }

    let extrap = extrapolatedTopNames(
        targetYear: year,
        nameYear: nameYear,
        periodYears: cyclePeriodYears,
        limit: limit
    )
    let names = extrap.map { (name: $0.name, value: $0.predictedTotal) }
    let detail = String(
        format: "Year %d is not in the file. Each name's past totals were fit in log space as "
            + "log(count+1) ~= w0 + w1*year + w2*sin(2*pi*y/%.0f) + w3*cos(2*pi*y/%.0f) when at least four distinct years exist (otherwise a log-linear trend in year). "
            + "Values shown are extrapolated total popularity scores, not observed counts.",
        cyclePeriodYears,
        cyclePeriodYears
    )
    return (names, .extrapolatedPattern, detail)
}

/// Core ML predicted `Count` for each sex at the given year (same for every name with that sex/year — useful as a baseline).
func predictedSexYearBaselines(forYear year: Int, model: MLModel) throws -> (female: Double, male: Double) {
    let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "Count"
    let fInput = try MLDictionaryFeatureProvider(dictionary: [
        "Year": MLFeatureValue(int64: Int64(year)),
        "Sex": MLFeatureValue(string: "F"),
    ])
    let mInput = try MLDictionaryFeatureProvider(dictionary: [
        "Year": MLFeatureValue(int64: Int64(year)),
        "Sex": MLFeatureValue(string: "M"),
    ])
    let fOut = try model.prediction(from: fInput)
    let mOut = try model.prediction(from: mInput)
    guard let female = fOut.featureValue(for: outputName)?.doubleValue,
          let male = mOut.featureValue(for: outputName)?.doubleValue
    else {
        throw BabyNamesRegressionError.trainingFailed("Could not read prediction output from the model.")
    }
    return (female, male)
}
