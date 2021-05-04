package poa.ml.image.objects.recognition

import poa.ml.image.objects.recognition.model.Sample
import smile.math.matrix.Matrix
import java.awt.Color
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Toolkit
import java.awt.geom.AffineTransform
import java.awt.geom.Area
import java.awt.image.BufferedImage
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.WindowConstants


fun showImage(image: BufferedImage, lambda: (JFrame) -> Unit = {}) {
    showJLabel(JLabel(ImageIcon(image)), lambda)
}

fun showJLabel(jLabel: JLabel, lambda: (JFrame) -> Unit = {}) {
    val frame = JFrame()
    val dim: Dimension = Toolkit.getDefaultToolkit().screenSize
    frame.contentPane.layout = FlowLayout()
    frame.contentPane.add(jLabel)
    frame.defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
    lambda(frame)
    frame.pack()
    frame.setLocation(dim.width / 2 - frame.size.width / 2, dim.height / 2 - frame.size.height / 2)
    frame.isVisible = true
}

fun resize(img: BufferedImage, targetHeight: Int): Pair<Double, BufferedImage> {
    val k = targetHeight.toDouble() / img.height.toDouble()
    val targetWidth = (img.width.toDouble() * k).toInt()
    val resizedImage = BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB)
    val graphics2D = resizedImage.createGraphics()
    graphics2D.drawImage(img, 0, 0, targetWidth, targetHeight, null)
    graphics2D.dispose()
    return k to resizedImage
}

fun BufferedImage.resized(targetHeight: Int) = resize(this, targetHeight).second

fun BufferedImage.clone(): BufferedImage {
    return resized(this.height)
}

fun Area.copy() = Area(this)

fun Area.scaled(scaleK: Double) = createTransformedArea(AffineTransform.getScaleInstance(1 / scaleK, 1 / scaleK))


fun toDoubleArray(image: BufferedImage): DoubleArray {
    val row = mutableListOf<Double>()
    for (y in 0 until image.height) {
        for (x in 0 until image.width) {
            val color = Color(image.getRGB(x, y))
            row.add(color.red.toDouble())
            row.add(color.green.toDouble())
            row.add(color.blue.toDouble())
        }
    }
    return row.toDoubleArray()
}

fun DoubleArray.scale(center: DoubleArray, scale: DoubleArray) =
    Matrix(arrayOf(this)).scale(center, scale).row(0)

fun highlightArea(sourceImage: BufferedImage, area: Area): BufferedImage {
    val res = sourceImage.clone()
    res.createGraphics().draw(area)
    return res
}
