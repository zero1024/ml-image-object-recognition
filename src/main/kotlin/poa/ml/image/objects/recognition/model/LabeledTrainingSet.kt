package poa.ml.image.objects.recognition.model

import poa.ml.image.objects.recognition.toDoubleArray
import smile.math.matrix.Matrix

class LabeledTrainingSet {

    private val rows = mutableListOf<DoubleArray>()
    private val labels = mutableListOf<Int>()

    fun add(samples: List<Pair<Sample, Boolean>>) {
        for ((sample, label) in samples) {
            labels.add(if (label) 1 else 0)
            val image = sample.image
            val array = toDoubleArray(image)
            rows.add(array)
        }
    }

    fun toMatrix(): Pair<Matrix, IntArray> {
        return Matrix(rows.toTypedArray()) to labels.toIntArray()
    }

}