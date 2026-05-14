//
//  Item.swift
//  AISampleApp
//
//  Created by Sean Kang on 5/14/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
