//
//  ContentView.swift
//  AISampleApp
//
//  Created by Sean Kang on 5/14/26.
//

import CoreML
import CreateML
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isImporting = false
    @State private var selectedFileURL: URL?
    @State private var selectedFileName = ""
    @State private var importMessage: String?

    @State private var isTraining = false
    @State private var trainingError: String?
    @State private var modelSummary = ""
    @State private var predictionLines: [String] = []
    @State private var trainedModel: MLModel?

    @State private var showYearPrompt = false
    @State private var yearField = ""
    @State private var isComputingTopNames = false
    @State private var queriedYear: Int?
    @State private var topNamesResults: [(name: String, totalCount: Double)] = []
    @State private var topNamesSource: BabyNamesYearResultSource?
    @State private var topNamesDetail = ""
    @State private var baselineFemale: Double?
    @State private var baselineMale: Double?
    @State private var topNamesError: String?

    private var canQueryYear: Bool {
        trainedModel != nil && selectedFileURL != nil && !isTraining
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)

                    Text("Baby names")
                        .font(.title2.weight(.semibold))
                        .frame(maxWidth: .infinity)

                    Text("Select babyNames.csv. The app trains a linear regression model on-device (Create ML) and runs predictions with the resulting Core ML model.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)

                    #if targetEnvironment(simulator)
                    Text("Note: Create ML training is intended for physical devices; Simulator runs may fail or differ.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    #endif

                    Button {
                        isImporting = true
                    } label: {
                        Label("Select File", systemImage: "folder")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isTraining)

                    if isTraining {
                        ProgressView("Training Core ML model…")
                            .frame(maxWidth: .infinity)
                    }

                    if !selectedFileName.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Selected file")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(selectedFileName)
                                .font(.body.monospaced())
                                .textSelection(.enabled)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    if let importMessage {
                        Text(importMessage)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    if let trainingError {
                        Text(trainingError)
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }

                    if trainedModel != nil {
                        Label("Core ML model ready", systemImage: "checkmark.circle.fill")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.green)
                    }

                    if !modelSummary.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Training summary")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(modelSummary)
                                .font(.footnote.monospaced())
                                .textSelection(.enabled)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    if !predictionLines.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Sample Core ML predictions")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            ForEach(Array(predictionLines.enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.footnote)
                                    .textSelection(.enabled)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    if let queriedYear {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Top 10 names in \(queriedYear)")
                                .font(.subheadline.weight(.semibold))
                            if let src = topNamesSource {
                                Text(src.rawValue)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            if topNamesResults.isEmpty {
                                Text("No ranking could be produced for this year.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            } else {
                                let valueHeader = topNamesSource == .extrapolatedPattern ? "Score" : "Count"
                                HStack {
                                    Text("#").frame(width: 24, alignment: .leading)
                                    Text("Name")
                                    Spacer()
                                    Text(valueHeader)
                                        .frame(minWidth: 44, alignment: .trailing)
                                }
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                ForEach(Array(topNamesResults.enumerated()), id: \.offset) { index, row in
                                    HStack {
                                        Text("\(index + 1).")
                                            .font(.footnote.monospacedDigit())
                                            .foregroundStyle(.secondary)
                                            .frame(width: 24, alignment: .leading)
                                        Text(row.name)
                                            .font(.footnote.weight(.medium))
                                        Spacer(minLength: 8)
                                        Text(String(format: "%.0f", row.totalCount))
                                            .font(.footnote.monospacedDigit())
                                            .foregroundStyle(.secondary)
                                            .frame(minWidth: 44, alignment: .trailing)
                                    }
                                }
                            }
                            if !topNamesDetail.isEmpty {
                                Text(topNamesDetail)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            if let f = baselineFemale, let m = baselineMale {
                                Text(
                                    String(
                                        format: "Core ML model (trained on your sample) at %d: F ≈ %.0f, M ≈ %.0f predicted count for year+sex rows.",
                                        queriedYear,
                                        f,
                                        m
                                    )
                                )
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    if let topNamesError {
                        Text(topNamesError)
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                }
                .padding()
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if canQueryYear {
                    VStack(spacing: 8) {
                        if isComputingTopNames {
                            ProgressView("Scanning CSV…")
                        }
                        Button {
                            yearField = queriedYear.map(String.init) ?? ""
                            topNamesError = nil
                            showYearPrompt = true
                        } label: {
                            Label("Predict top 10 names for a year…", systemImage: "calendar")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding(.horizontal)
                    .padding(.top, 10)
                    .padding(.bottom, 8)
                    .background(.bar)
                }
            }
            .navigationTitle("AISampleApp")
            .fileImporter(
                isPresented: $isImporting,
                allowedContentTypes: [.commaSeparatedText, .plainText],
                allowsMultipleSelection: false
            ) { result in
                importMessage = nil
                trainingError = nil
                modelSummary = ""
                predictionLines = []
                trainedModel = nil
                queriedYear = nil
                topNamesResults = []
                topNamesSource = nil
                topNamesDetail = ""
                baselineFemale = nil
                baselineMale = nil
                topNamesError = nil

                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    selectedFileURL = url
                    selectedFileName = url.lastPathComponent
                    if url.lastPathComponent.lowercased() == "babynames.csv" {
                        importMessage = "Training on-device from babyNames.csv…"
                    } else {
                        importMessage = "Expected babyNames.csv; training anyway if the format matches."
                    }
                    Task { await trainFromSelectedFile(url: url) }
                case .failure(let error):
                    selectedFileURL = nil
                    selectedFileName = ""
                    importMessage = error.localizedDescription
                }
            }
            .alert("Year for top names", isPresented: $showYearPrompt) {
                TextField("e.g. 2010", text: $yearField)
                    .keyboardType(.numberPad)
                Button("Cancel", role: .cancel) {}
                Button("Predict") {
                    Task { await computeTopNamesForYear() }
                }
            } message: {
                Text("If the year appears in your CSV, rankings use observed counts. If not, the app extrapolates each name’s history with a log trend plus a long-period cycle, and still runs your Core ML year+sex model for baselines.")
            }
        }
    }

    @MainActor
    private func trainFromSelectedFile(url: URL) async {
        isTraining = true
        trainingError = nil
        modelSummary = ""
        predictionLines = []
        trainedModel = nil
        queriedYear = nil
        topNamesResults = []
        topNamesSource = nil
        topNamesDetail = ""
        baselineFemale = nil
        baselineMale = nil
        topNamesError = nil
        defer { isTraining = false }

        do {
            let outcome = try await Task.detached(priority: .userInitiated) {
                try trainBabyNamesCountRegressor(from: url)
            }.value
            trainedModel = outcome.model
            modelSummary = outcome.summary
            predictionLines = outcome.sampleComparisons
            importMessage = "Training finished. Predictions use Core ML’s MLModel.prediction(from:)."
        } catch {
            trainingError = error.localizedDescription
            importMessage = nil
        }
    }

    @MainActor
    private func computeTopNamesForYear() async {
        topNamesError = nil
        guard let url = selectedFileURL, let model = trainedModel else {
            topNamesError = "Train a model first by choosing a CSV file."
            return
        }
        let trimmed = yearField.trimmingCharacters(in: .whitespaces)
        guard let year = Int(trimmed), (1800 ... 2100).contains(year) else {
            topNamesError = "Enter a valid year between 1800 and 2100."
            return
        }

        isComputingTopNames = true
        defer { isComputingTopNames = false }

        do {
            let resolved = try await Task.detached(priority: .userInitiated) {
                try topBabyNamesExactOrExtrapolated(forYear: year, csvURL: url, cyclePeriodYears: 40, limit: 10)
            }.value
            let baseline = try predictedSexYearBaselines(forYear: year, model: model)
            queriedYear = year
            topNamesResults = resolved.names.map { (name: $0.name, totalCount: $0.value) }
            topNamesSource = resolved.source
            topNamesDetail = resolved.detail
            baselineFemale = baseline.female
            baselineMale = baseline.male
            showYearPrompt = false
        } catch {
            topNamesError = error.localizedDescription
        }
    }
}

#Preview {
    ContentView()
}
