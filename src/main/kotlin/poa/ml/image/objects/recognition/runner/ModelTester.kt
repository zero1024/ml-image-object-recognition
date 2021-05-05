package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.printlnEnd
import poa.ml.image.objects.recognition.printlnStart
import poa.ml.image.objects.recognition.readFromFile
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.mlp
import smile.math.matrix.Matrix

class ModelTester {

    fun test(trainingSetFile: String) {
        printlnStart("===Reading the matrix...")
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        printlnEnd("===Done reading. Size = ${X.size()}")
        printlnStart("===Reading the labels...")
        val y = readFromFile<IntArray>("$trainingSetFile.y")
        printlnEnd("===Done reading.")

        val (center, scale) = readFromFile<Pair<DoubleArray, DoubleArray>>("$trainingSetFile.options")

        val x = X.toArray()
        printlnStart("===Training...")
        val classifier =
            mlp(x, y, arrayOf(Layer.sigmoid(18), Layer.sigmoid(9), Layer.mle(1, OutputFunction.SIGMOID)))
        printlnEnd("===Done training.")
    }

}