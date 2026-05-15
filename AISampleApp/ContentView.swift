//
//  ContentView.swift
//  AISampleApp
//
//  Created by Sean Kang on 5/14/26.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isImporting = false
    @State private var selectedFileURL: URL?
    @State private var selectedFileName = ""
    @State private var importMessage: String?

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Image(systemName: "doc.text")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)

                Text("Baby names")
                    .font(.title2.weight(.semibold))

                Text("Select babyNames.csv from Files or On My iPhone.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                Button {
                    isImporting = true
                } label: {
                    Label("Select File", systemImage: "folder")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                }
                .buttonStyle(.borderedProminent)

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

                Spacer()
            }
            .padding()
            .navigationTitle("AISampleApp")
            .fileImporter(
                isPresented: $isImporting,
                allowedContentTypes: [.commaSeparatedText, .plainText],
                allowsMultipleSelection: false
            ) { result in
                importMessage = nil
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    selectedFileURL = url
                    selectedFileName = url.lastPathComponent
                    if url.lastPathComponent.lowercased() == "babynames.csv" {
                        importMessage = "babyNames.csv is ready to use."
                    } else {
                        importMessage = "For this sample, choose a file named babyNames.csv."
                    }
                case .failure(let error):
                    selectedFileURL = nil
                    selectedFileName = ""
                    importMessage = error.localizedDescription
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
