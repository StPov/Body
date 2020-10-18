//
//  MainViewController + Processing.swift
//  BodyM
//
//  Created by Stanislav Povolotskiy on 14.10.2020.
//  Copyright © 2020 Stanislav Povolotskiy. All rights reserved.
//

import Foundation
import Vision

extension MainViewController {
    // MARK: - Setup Core ML
    func setUpModel(model: MLModel, isSegmenting: Bool) {
        self.isSegmenting = isSegmenting
        if isSegmenting {
            if let visionModel = try? VNCoreMLModel(for: model) {
                self.visionModelSeg = visionModel
                requestSeg = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                    self.segment(request: request)
                })
                requestSeg?.imageCropAndScaleOption = .scaleFill
            } else {
                fatalError("cannot load the ml model")
            }
        } else {
            if let visionModel = try? VNCoreMLModel(for: model) {
                self.visionModel = visionModel
                request = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                    self.detectKeyPoints(request: request)
                })
                request?.imageCropAndScaleOption = .scaleFill
            } else {
                fatalError("cannot load the ml model")
            }
        }
    }
    // MARK: - Inference
    func predict(with url: URL) {
        guard let request1 = requestSeg else { fatalError() }
        guard let request2 = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(url: url, options: [:])
        try? handler.perform([request1, request2])
    }
}
