package poa.ml.image.objects.recognition

import org.junit.jupiter.api.Test
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
    internal fun testMlpLayers() {
        val plot = mutableListOf<DoubleArray>()
        for (i in 3..60 step 3) {
            val validation = modelTrainer.test("/Users/oleg1024/Downloads/divan/heart", "mlp_$i", 3000)
            plot.add(doubleArrayOf(i.toDouble(), validation.avg.accuracy))
            linePlot(plot.toTypedArray())
        }
        linePlot(plot.toTypedArray())
        Thread.sleep(1000000)
    }

    @Test
    internal fun testMlpWitCv() {

        val trainingSetFile = "/Users/oleg1024/Downloads/divan/heart"
        val (X, y) = modelTrainer.getTrainingSet(trainingSetFile)

        val errorsStat = mutableListOf<Pair<DoubleArray, String>>()

        val (trX, trY) = subSet(X, y, to = 21000)
        val (cvX, cvY) = subSet(X, y, from = 21001, to = 26000)

        (2000..20000 step 1000)
            .forEach { iter ->
                printlnStart("Iteration $iter")

                val (iterX, iterY) = subSet(trX, trY, to = iter)

                val array = iterX.toArray()

                val mlp = mlp(
                    array,
                    iterY,
                    arrayOf(Layer.sigmoid(12), Layer.mle(2, OutputFunction.SIGMOID)),
                    epochs = 500
                )

                val predictions = mlp.predict(array)
                val misclassError = predictions.add(iterY).filter { it == 1 }.size
                val trPoint = doubleArrayOf(iter.toDouble(), misclassError.toDouble())
                errorsStat.add(trPoint to "training")

                val cvPredictions = mlp.predict(cvX.toArray())
                val cvMisclassError = cvPredictions.add(cvY).filter { it == 1 }.size
                val cvPoint = doubleArrayOf(iter.toDouble(), cvMisclassError.toDouble())
                errorsStat.add(cvPoint to "cv")
                printlnEnd("Done.")
                scatterPlot(errorsStat.map { it.first }.toTypedArray(), errorsStat.map { it.second }.toTypedArray())
            }

        scatterPlot(errorsStat.map { it.first }.toTypedArray(), errorsStat.map { it.second }.toTypedArray())
        Thread.sleep(100000000)
    }

    @Test
    internal fun testMlpConvergence() {

        val nrows = 5000

        val trainingSetFile = "/Users/oleg1024/Downloads/divan/heart"
        val (X, y) = modelTrainer.getTrainingSet(trainingSetFile)

        val errorsStat = mutableListOf<Pair<DoubleArray, String>>()

        val (trX, trY) = subSet(X, y, to = nrows)

        val array = trX.toArray()
        val mlp = mlp(array, y, arrayOf(Layer.sigmoid(12), Layer.mle(2, OutputFunction.SIGMOID)), epochs = 0)

        (0..600)
            .forEach { iter ->
                printlnStart("Iteration $iter")
                mlp.update(array, trY)

                val predictions = mlp.predict(array)
                val misclassError = predictions.add(trY).filter { it == 1 }.size
                val trPoint = doubleArrayOf(iter.toDouble(), misclassError.toDouble())
                errorsStat.add(trPoint to "training")

                printlnEnd("Done.")
            }

        scatterPlot(errorsStat.map { it.first }.toTypedArray(), errorsStat.map { it.second }.toTypedArray())
        Thread.sleep(100000000)
    }

    @Test
    internal fun testWithDir() {

        val trainingSetFile = "/Users/oleg1024/Downloads/iris/iris"

        val classifier = modelTrainer.train(trainingSetFile, "mlp_12", 12000)

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
