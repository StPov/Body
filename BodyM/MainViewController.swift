import UIKit
import Vision
import SwiftImage

class MainViewController: UIViewController {

    @IBOutlet weak var drawingView: DrawingSegmentationView!
    @IBOutlet private weak var segmentedControl: UISegmentedControl!
    @IBOutlet private weak var textField: UITextField!
    @IBOutlet private weak var tableView: UITableView!
    @IBOutlet private weak var imageView: UIImageView!
    @IBOutlet private weak var photosCountLabel: UILabel!
    @IBOutlet private weak var calculateButton: UIButton!
    @IBOutlet private weak var tableViewHeightConstraint: NSLayoutConstraint!
    
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
    var frontSegmentationmap: SegmentationResultMLMultiArray? = nil
    var sideSegmentationmap: SegmentationResultMLMultiArray? = nil
    
    let imagePickerController = UIImagePickerController()
    var isSegmenting = true
    var isFrontImageMode = true
    var isFrontImageChosen = false
    var isSideImageChosen = false
    var frontImage = UIImage()
    var sideImage = UIImage()
    var frontSegmentedImage = UIImage()
    var sideSegmentedImage = UIImage()
    var photosCount = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setUpModel(model: segmentationModel.model, isSegmenting: true)
        setUpModel(model: posenetModel.model, isSegmenting: false)
        
        imagePickerController.delegate = self
        
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(imageTapped(tapGestureRecognizer:)))
        imageView.isUserInteractionEnabled = true
        imageView.addGestureRecognizer(tapGestureRecognizer)
        drawingView.isUserInteractionEnabled = true
        drawingView.addGestureRecognizer(tapGestureRecognizer)
        
        imageView.image = UIImage(named: "front_body")
        photosCountLabel.text = "0/0 done"
        photosCountLabel.textColor = .black
        calculateButton.isEnabled = false
        tableView.isHidden = true
    }
    
    @objc func imageTapped(tapGestureRecognizer: UITapGestureRecognizer) {
        self.present(imagePickerController, animated: true)
    }
    
    @IBAction func indexChanged(_ sender: Any) {
        switch segmentedControl.selectedSegmentIndex {
        case 0:
            isFrontImageMode = true
            if isFrontImageChosen {
                imageView.image = frontImage
            } else {
                imageView.image = UIImage(named: "front_body")
            }
            drawingView.segmentationmap = frontSegmentationmap //drows here to see separate segmentation for correspondant photo
        case 1:
            isFrontImageMode = false
            if isSideImageChosen {
                imageView.image = sideImage
            } else {
                imageView.image = UIImage(named: "side_body")
            }
            drawingView.segmentationmap = sideSegmentationmap
        default:
            break
        }
    }
    
    @IBAction func calculateButtonPressed(_ sender: UIButton) {
        tableView.isHidden = false
    }
}

// MARK: - UIImagePickerControllerDelegate & UINavigationControllerDelegate
extension MainViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if isFrontImageMode {
            if let image = info[.originalImage] as? UIImage,
                let url = info[.imageURL] as? URL {
                imageView.image = image
                frontImage = image
                isFrontImageChosen = true
                self.predict(with: url)
                print("Front image size: \(frontSegmentedImage.size.height) per \(frontSegmentedImage.size.width)")
            }
        } else {
            if let image = info[.originalImage] as? UIImage,
                let url = info[.imageURL] as? URL {
                imageView.image = image
                sideImage = image
                isSideImageChosen = true
                self.predict(with: url)
                print("Side image size: \(sideSegmentedImage.size.height) per \(sideSegmentedImage.size.width)")
            }
        }
        photosCount += 1
        photosCountLabel.text = "\(photosCount)/2 done"
        if photosCount >= 2 {
            photosCountLabel.textColor = .systemGreen
            calculateButton.isEnabled = true
        }
        dismiss(animated: true, completion: nil)
    }
}

