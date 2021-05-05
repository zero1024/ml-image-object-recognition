package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.printlnEnd
import poa.ml.image.objects.recognition.printlnStart
import poa.ml.image.objects.recognition.readFromFile
import smile.base.mlp.Layer
import smile.base.mlp.LayerBuilder
import smile.base.mlp.OutputFunction
import smile.classification.Classifier
import smile.classification.logit
import smile.classification.mlp
import smile.math.matrix.Matrix
import smile.validation.CrossValidation
import java.util.function.BiFunction

class ModelTester {

    fun test(
        trainingSetFile: String,
        classifierCode: String
    ) {
        printlnStart("===Reading the matrix...")
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        printlnEnd("===Done reading. Size = ${X.nrows()}x${X.ncols()}")
        printlnStart("===Reading the labels...")
        val y = readFromFile<IntArray>("$trainingSetFile.y")
        printlnEnd("===Done reading.")

        printlnStart("===Training $classifierCode classifier...")
        val classification = CrossValidation.classification(5, X.toArray(), y, dispatch(classifierCode))
        println("")
        println("")
        println("")
        println(classification.avg)
        println("")
        println("")
        println("")
        printlnEnd("===Done training.")
    }

}


fun dispatch(name: String): BiFunction<Array<DoubleArray>, IntArray, Classifier<DoubleArray>> {
    if (name.startsWith("mlp.")) {
        val layers = name.split(".")
            .mapNotNull { it.toIntOrNull() }
            .map { Layer.sigmoid(it) as LayerBuilder }
            .toMutableList()
        layers.add(Layer.mle(1, OutputFunction.SIGMOID))
        return BiFunction { X, y -> mlp(X, y, layers.toTypedArray()) }
    } else if (name.startsWith("logit.")) {
        val lambda = name.split("logit.")[1]
        return BiFunction { X, y -> logit(X, y, lambda.toDouble()) }
    } else {
        throw IllegalStateException()
    }
}
