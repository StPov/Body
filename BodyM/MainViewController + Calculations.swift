//
//  Calculations.swift
//  BodyM
//
//  Created by Stanislav Povolotskiy on 14.10.2020.
//  Copyright © 2020 Stanislav Povolotskiy. All rights reserved.
//

import UIKit
import Vision

typealias KeyPoint = (name: KeyPoints, point: CGPoint)
enum KeyPoints: String {
    case leftForearm = "lWrist-lElbow"
    case leftUpperArm = "lElbow-lShoulder"
    case rightForearm = "rWrist-rElbow"
    case rightUpperArm = "rElbow-rShoulder"
    case middleBody = "shoulders-hips"
    case leftThigh = "lHip-lKnee"
    case leftCalf = "lKnee-lAnkle"
    case rightThigh = "rHip-rKnee"
    case rightCalf = "rKnee-rAnkle"
}

extension MainViewController {
    func segment(request: VNRequest) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let segmentationmap = observations.first?.featureValue.multiArrayValue {
            
            
            if isFrontImageMode {
                frontSegmentationmap = SegmentationResultMLMultiArray(mlMultiArray: segmentationmap)
                drawingView.segmentationmap = frontSegmentationmap //drows here to see separate segmentation for  photo immediately
                frontSegmentedImage = drawingView.asImage()
            } else {
                sideSegmentationmap = SegmentationResultMLMultiArray(mlMultiArray: segmentationmap)
                drawingView.segmentationmap = sideSegmentationmap
                sideSegmentedImage = drawingView.asImage()
            }
        }
    }
    
    func detectKeyPoints(request: VNRequest) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let heatmaps = observations.first?.featureValue.multiArrayValue {

            /* ========================= post-processing ========================= */

            /* ------------------ convert heatmap to point array ----------------- */
            var predictedPoints = postProcessor.convertToPredictedPoints(from: heatmaps)

            /* --------------------- moving average filter ----------------------- */
            if predictedPoints.count != mvfilters.count {
                mvfilters = predictedPoints.map { _ in MovingAverageFilter(limit: 3) }
            }
            for (predictedPoint, filter) in zip(predictedPoints, mvfilters) {
                filter.add(element: predictedPoint)
            }
            predictedPoints = mvfilters.map { $0.averagedValue() }
            
            guard let middleKeyPoints = getMiddleKeyPoints(predictedPoints: predictedPoints) else { return }
        }
    }
    
    func getMiddleKeyPoints(predictedPoints: [PredictedPoint?]) -> [KeyPoint]? {
            
        guard let top = predictedPoints[0]?.maxPoint,
            let neck = predictedPoints[1]?.maxPoint,
            let rShoulder = predictedPoints[2]?.maxPoint,
            let rElbow = predictedPoints[3]?.maxPoint,
            let rWrist = predictedPoints[4]?.maxPoint,
            let lShoulder = predictedPoints[5]?.maxPoint,
            let lElbow = predictedPoints[6]?.maxPoint,
            let lWrist = predictedPoints[7]?.maxPoint,
            let rHip = predictedPoints[8]?.maxPoint,
            let rKnee = predictedPoints[9]?.maxPoint,
            let rAnkle = predictedPoints[10]?.maxPoint,
            let lHip = predictedPoints[11]?.maxPoint,
            let lKnee = predictedPoints[12]?.maxPoint,
            let lAnkle = predictedPoints[13]?.maxPoint else { return nil}
        
        let keyPoints = [(KeyPoints.leftForearm, findMiddlePoint(between: lWrist, and: lElbow)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: lElbow, and: lShoulder)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: rWrist, and: rElbow)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: rElbow, and: rShoulder)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: findMiddlePoint(between: lShoulder, and: rShoulder), and: findMiddlePoint(between: lHip, and: rHip))),
                         (KeyPoints.leftForearm, findMiddlePoint(between: lHip, and: lKnee)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: lKnee, and: lAnkle)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: rHip, and: rKnee)),
                         (KeyPoints.leftForearm, findMiddlePoint(between: rKnee, and: rAnkle))]
        return keyPoints
    }
    
    func findMiddlePoint(between point1: CGPoint, and point2: CGPoint) -> CGPoint{
        return CGPoint(x: (point1.x + point2.x) / 2, y: (point1.y + point2.y) / 2)
    }
}
