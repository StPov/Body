import UIKit
import Vision
import SwiftImage

class StillImageViewController: UIViewController {

    // MARK: - UI Properties
    @IBOutlet weak var mainImageView: UIImageView!
    @IBOutlet weak var drawingView: DrawingSegmentationView!
    
    let imagePickerController = UIImagePickerController()
    
    // MARK - Core ML model
    // DeepLabV3(iOS12+), DeepLabV3FP16(iOS12+), DeepLabV3Int8LUT(iOS12+)
    let segmentationModel = DeepLabV3()
    let posenetModel = model_cpm()
    
    // MARK: - Vision Properties
    var requestSeg: VNCoreMLRequest?
    var visionModelSeg: VNCoreMLModel?
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    
    // Postprocess
    var postProcessor: HeatmapPostProcessor = HeatmapPostProcessor()
    var mvfilters: [MovingAverageFilter] = []
    
    var isSegmenting = true
        
    override func viewDidLoad() {
        super.viewDidLoad()

        setUpModel(model: segmentationModel.model, isSegmenting: true)
        setUpModel(model: posenetModel.model, isSegmenting: false)
        
        imagePickerController.delegate = self
    }
    
    @IBAction func tapCamera(_ sender: Any) {
        self.present(imagePickerController, animated: true)
    }
    
    // MARK: - Setup Core ML
    func setUpModel(model: MLModel, isSegmenting: Bool) {
        self.isSegmenting = isSegmenting
        if isSegmenting {
            if let visionModel = try? VNCoreMLModel(for: model) {
                self.visionModelSeg = visionModel
                requestSeg = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                    self.blo(request: request)
                })
                requestSeg?.imageCropAndScaleOption = .scaleFill
            } else {
                fatalError("cannot load the ml model")
            }
        } else {
            if let visionModel = try? VNCoreMLModel(for: model) {
                self.visionModel = visionModel
                request = VNCoreMLRequest(model: visionModel, completionHandler: { (request, error) in
                    self.bla(request: request)
                })
                request?.imageCropAndScaleOption = .scaleFill
            } else {
                fatalError("cannot load the ml model")
            }
        }
    }
}

// MARK: - Inference
extension StillImageViewController {
    func predict(with url: URL) {
        guard let request1 = requestSeg else { fatalError() }
        guard let request2 = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(url: url, options: [:])
        try? handler.perform([request1, request2])
        
    }
}

// MARK: - UIImagePickerControllerDelegate & UINavigationControllerDelegate
extension StillImageViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let image = info[.originalImage] as? UIImage,
            let url = info[.imageURL] as? URL {
            mainImageView.image = image
            self.predict(with: url)
        }
        dismiss(animated: true, completion: nil)
    }
}


extension StillImageViewController {
    func blo(request: VNRequest) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let segmentationmap = observations.first?.featureValue.multiArrayValue {
            
            drawingView.segmentationmap = SegmentationResultMLMultiArray(mlMultiArray: segmentationmap)
        }
    }
    
    func bla(request: VNRequest) {
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
            
            getMeasures(realHeight: 182, predictedPoints: predictedPoints)
        }
    }
    
    func findDistance(between point1: CGPoint, point2: CGPoint) -> CGFloat{
        let a = pow(point2.x - point1.x, 2)
        let b = pow(point2.y - point1.y, 2)
        let d = sqrtf(Float(a + b))
        return CGFloat(d)
    }
    
    func findMiddlePoint(onePoint: CGPoint, anotherPoint: CGPoint) -> CGPoint{
        return CGPoint(x: (onePoint.x + anotherPoint.x) / 2, y: (onePoint.y + anotherPoint.y) / 2)
    }
    
    func getMeasures(realHeight: CGFloat, predictedPoints: [PredictedPoint?]) -> [(type: String, distance: CGFloat)]? {
            
        guard let topPointY = predictedPoints[0]?.maxPoint.y, let leftAnklePointY = predictedPoints[13]?.maxPoint.y, let rightAnklePointY = predictedPoints[10]?.maxPoint.y else { return nil}
        let bottomPointY = leftAnklePointY >= rightAnklePointY ? leftAnklePointY : rightAnklePointY
            
        let systemHeight = bottomPointY - topPointY
        let k = realHeight / systemHeight
            
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
        
        var bla = [(type: String, dist: CGFloat)]()
            
        let distTopNeck = findDistance(between: top, point2: neck)
//        bla.append()
        let distRShoulderRElbow = findDistance(between: rShoulder, point2: rElbow)
        let distRElbowRWrist = findDistance(between: rElbow, point2: rWrist)
        let distLShoulderLElbow = findDistance(between: lShoulder, point2: lElbow)
        let distLElbowLWrist = findDistance(between: lElbow, point2: lWrist)
        let distRHipRKnee = findDistance(between: rHip, point2: rKnee)
        let distRKneeRAnkle = findDistance(between: rKnee, point2: rAnkle)
        let distLHipLKnee = findDistance(between: lHip, point2: lKnee)
        let distLKneeLAnkle = findDistance(between: lKnee, point2: lAnkle)
        return nil
    }
}
