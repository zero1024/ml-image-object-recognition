package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.printlnEnd
import poa.ml.image.objects.recognition.printlnStart
import poa.ml.image.objects.recognition.readFromFile
import smile.base.mlp.Layer
import smile.base.mlp.LayerBuilder
import smile.base.mlp.OutputFunction
import smile.classification.Classifier
import smile.classification.mlp
import smile.math.matrix.Matrix

class ModelTester {

    fun test(
        trainingSetFile: String,
        classifier: String
    ) {
        printlnStart("===Reading the matrix...")
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        printlnEnd("===Done reading. Size = ${X.nrows()}x${X.ncols()}")
        printlnStart("===Reading the labels...")
        val y = readFromFile<IntArray>("$trainingSetFile.y")
        printlnEnd("===Done reading.")

        val (center, scale) = readFromFile<Pair<DoubleArray, DoubleArray>>("$trainingSetFile.options")

        val xArray = X.submatrix(0, 0, 499, X.ncols() - 1).toArray()
        val yArray = y.copyOf(500)

        printlnStart("===Training $classifier classifier...")
        val classifier = dispatch(classifier)(X.toArray(), y)
        printlnEnd("===Done training.")
        println(1)
    }

}


fun dispatch(name: String): (Array<DoubleArray>, IntArray) -> Classifier<DoubleArray> {
    if (name.startsWith("mlp.")) {
        val layers = name.split(".")
            .mapNotNull { it.toIntOrNull() }
            .map { Layer.sigmoid(it) as LayerBuilder }
            .toMutableList()
        layers.add(Layer.mle(1, OutputFunction.SIGMOID))
        return { X, y -> mlp(X, y, layers.toTypedArray()) }
    } else {
        throw IllegalStateException()
    }
}
