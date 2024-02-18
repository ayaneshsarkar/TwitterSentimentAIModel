import Cocoa
import CreateML

let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/ayaneshsarkar/Downloads/twitter-sanders-apple3.csv"))

let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")
let evaluationMetrics = sentimentClassifier.evaluation(on: trainingData, textColumn: "text", labelColumn: "class")

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metadata = MLModelMetadata(author: "Ayanesh Sarkar", shortDescription: "Making Dataset Model", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/ayaneshsarkar/Downloads/TweetSentimentClassifier.mlmodel"))

try sentimentClassifier.prediction(from: "@Apple is a terrible company!")

try sentimentClassifier.prediction(from: "I just found the best resurant ever, and it's @DuckAndWaffle")

try sentimentClassifier.prediction(from: "I think CocaCola ads are just ok.")
