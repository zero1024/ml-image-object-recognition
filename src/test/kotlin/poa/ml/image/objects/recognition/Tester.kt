package poa.ml.image.objects.recognition

import org.junit.jupiter.api.Test
import poa.ml.image.objects.recognition.deeplearning4j.Dl4jNNClassifier
import poa.ml.image.objects.recognition.runner.ModelTrainer
import poa.ml.image.objects.recognition.runner.TrainingSetCollector
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.mlp
import java.awt.Toolkit

class Tester {

    private val modelTrainer = ModelTrainer()
    private val samplesCollector = ImageSamplesCollector(
        pxlStep = 50,
        slideSize = 60
    )

    @Test
    internal fun testRotate() {
        val rotated = rotate90(
            arrayOf(
                doubleArrayOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
            ), 3
        )
        assert(rotated[0].contentEquals(doubleArrayOf(6.0, 3.0, 0.0, 7.0, 4.0, 1.0, 8.0, 5.0, 2.0)))
    }

    @Test
    internal fun train() {
        TrainingSetCollector().train("/Users/oleg1024/Downloads/iris/", "/Users/oleg1024/Downloads/iris/iris")
    }


    @Test
    internal fun testLogitWithRespectToTheSizeOfTheInput() {
        val (X, y) = modelTrainer.getTrainingSet("/Users/oleg1024/Downloads/divan/heart")
        val f1 = mutableListOf<DoubleArray>()
        val accuracy = mutableListOf<DoubleArray>()
        for (i in 50..500 step 10) {
            val (subX, subY) = subSet(X, y, to = i)
            val classification = modelTrainer.test(subX, subY, "logit_0.01")
            f1.add(doubleArrayOf(i.toDouble(), classification.avg.f1))
            accuracy.add(doubleArrayOf(i.toDouble(), classification.avg.accuracy))
        }
        linePlot(f1.toTypedArray())
        linePlot(accuracy.toTypedArray())
        Thread.sleep(1000000)
    }

    @Test
    internal fun testLogit() {
        modelTrainer.test("/Users/oleg1024/Downloads/iris/iris", "logit_0")
        modelTrainer.test("/Users/oleg1024/Downloads/iris/iris", "logit_0.01") //the best
        modelTrainer.test("/Users/oleg1024/Downloads/iris/iris", "logit_0.03")
    }

    @Test
    internal fun testMlp() {
        modelTrainer.test("/Users/oleg1024/Downloads/iris/iris", "mlp_12")
    }

    @Test
    internal fun testDl4j() {
        modelTrainer.test("/Users/oleg1024/Downloads/iris/iris", "dl4j")
    }

    @Test
    internal fun testDl4jWitCv() {

        val trainingSetFile = "/Users/oleg1024/Downloads/divan/heart"
        val (X, y) = modelTrainer.getTrainingSet(trainingSetFile)

        val errorsStat = mutableListOf<Pair<DoubleArray, String>>()

        val (trX, trY) = subSet(X, y, to = 21000)
        val (cvX, cvY) = subSet(X, y, from = 21001, to = 23000)

        (100..5000 step 200)
            .forEach { iter ->
                printlnStart("Iteration $iter")

                val (iterX, iterY) = subSet(trX, trY, to = iter)

                val array = iterX.toArray()

                val classifier = Dl4jNNClassifier(array, iterY)

                val predictions = classifier.predict(array)
                val misclassError = predictions.add(iterY).filter { it == 1 }.size
                val trPoint = doubleArrayOf(iter.toDouble(), misclassError.toDouble())
                errorsStat.add(trPoint to "training")

                val cvPredictions = classifier.predict(cvX.toArray())
                val cvMisclassError = cvPredictions.add(cvY).filter { it == 1 }.size
                val cvPoint = doubleArrayOf(iter.toDouble(), cvMisclassError.toDouble())
                errorsStat.add(cvPoint to "cv")
                printlnEnd("Done.")
            }

        scatterPlot(errorsStat.map { it.first }.toTypedArray(), errorsStat.map { it.second }.toTypedArray())
        Thread.sleep(100000000)
    }

    @Test
    internal fun testDl4jConvergence() {

        val trainingSetFile = "/Users/oleg1024/Downloads/divan/heart"
        val (X, y) = modelTrainer.getTrainingSet(trainingSetFile)

        val errorsStat = mutableListOf<DoubleArray>()

        val classifier = Dl4jNNClassifier(X.toArray(), y)
        for ((idx, iter) in classifier.scores.listIteration.elements().withIndex()) {
            val score = classifier.scores.listScore.elements()[idx]
            errorsStat.add(doubleArrayOf(iter.toDouble(), score))
        }

        linePlot(errorsStat.toTypedArray())
        Thread.sleep(100000000)
    }

    @Test
    internal fun testWithDir() {

        val trainingSetFile = "/Users/oleg1024/Downloads/iris/iris"

        val classifier = modelTrainer.train(trainingSetFile, "dl4j", 16000)

        val centerAndScale = readFromFile<Pair<DoubleArray, DoubleArray>>("$trainingSetFile.options")
        val (center, scale) = centerAndScale

        val images = walkFileTree("/Users/oleg1024/Downloads/iris/")


        for (image in images) {
            val (scaleK, testImage) = resize(image, targetHeight = 400)
            val samples = samplesCollector.collect(testImage)
            val areaSums = samples
                .map { toDoubleArray(it.image).scale(center, scale) to it }
                .map { (array, sample) -> classifier.predict(array) to sample }
                .filter { (label, _) -> label == 1 }
                .map { (_, sample) -> sample.toArea() }
                .let { AreaSums(it) }

            val screenSize = Toolkit.getDefaultToolkit().screenSize.height.toDouble() * 0.8
            var (anotherScaleK, resultImage) = resize(image, screenSize.toInt())
            for (area in areaSums) {
                resultImage = highlightArea(resultImage, area.scaled(scaleK / anotherScaleK))
            }
            showImage(resultImage)
        }

        Thread.sleep(100000000)

    }


}
