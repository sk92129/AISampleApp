//
//  BabyNamesYearResultSource.swift
//  AISampleApp
//
//  Shared result label for year queries (exact CSV vs pattern extrapolation).
//

import Foundation

enum BabyNamesYearResultSource: String {
    case exactFromCSV = "Exact (from CSV)"
    case extrapolatedPattern = "Predicted (pattern extrapolation)"
}
