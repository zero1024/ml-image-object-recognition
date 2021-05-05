package poa.ml.image.objects.recognition.model

import poa.ml.image.objects.recognition.toDoubleArray
import smile.math.matrix.Matrix
import java.io.Serializable
import java.util.concurrent.CopyOnWriteArrayList


class LabeledTrainingSet(
    private val rows: CopyOnWriteArrayList<Pair<DoubleArray, Int>>
) : Serializable {

    constructor() : this(CopyOnWriteArrayList<Pair<DoubleArray, Int>>())

    fun size() = rows.size

    fun positiveLabelsPercentage(): Int {
        val positive = rows.filter { it.second == 1 }.size
        return (positive * 100) / size()
    }

    fun add(samples: List<Pair<Sample, Boolean>>) {
        for ((sample, label) in samples) {
            val image = sample.image
            val array = toDoubleArray(image)
            rows.add(array to if (label) 1 else 0)
        }
    }

    fun shuffle() = rows.shuffle()

    fun toMatrix(): Pair<Matrix, IntArray> {
        val X = rows.map { it.first }.toTypedArray()
        val y = rows.map { it.second }.toIntArray()
        return Matrix(X) to y
    }

    fun toMatrix(from: Int = 0, size: Int): Pair<Matrix, IntArray> {
        val subList = rows.subList(from, size)
        val X = subList.map { it.first }.toTypedArray()
        val y = subList.map { it.second }.toIntArray()
        return Matrix(X) to y
    }
}
