package poa.ml.image.objects.recognition

import org.apache.commons.io.FileUtils
import org.apache.commons.lang3.SerializationUtils
import smile.math.matrix.Matrix
import smile.plot.swing.LinePlot
import java.awt.Color
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Toolkit
import java.awt.geom.AffineTransform
import java.awt.geom.Area
import java.awt.image.BufferedImage
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.Serializable
import java.lang.management.ManagementFactory
import java.nio.file.FileVisitResult
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.SimpleFileVisitor
import java.nio.file.attribute.BasicFileAttributes
import javax.imageio.ImageIO
import javax.swing.*


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

fun walkFileTree(path: String): List<BufferedImage> {
    val res = mutableListOf<BufferedImage>()
    walkFileTree(path) { res.add(it) }
    return res
}

fun walkFileTree(path: String, lambda: (BufferedImage) -> Unit) {
    Files.walkFileTree(Path.of(path), object : SimpleFileVisitor<Path>() {
        override fun visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult {
            if (file.toFile().name.isImage()) {
                val image = ImageIO.read(file.toFile())
                lambda(image)
            }
            return FileVisitResult.CONTINUE
        }

        private fun String.isImage() = endsWith(".jpg") || endsWith(".png") || endsWith(".jpeg")
    })
}

fun memoryUsed(): String {
    val memoryMXBean = ManagementFactory.getMemoryMXBean()
    return FileUtils.byteCountToDisplaySize(memoryMXBean.heapMemoryUsage.used)
}

fun Serializable.saveToFile(file: String) {
    FileOutputStream(file).use {
        SerializationUtils.serialize(this, it)
    }
}

fun <T : Serializable> readFromFile(file: String): T {
    return BufferedInputStream(
        ProgressMonitorInputStream(
            JFrame(),
            "Reading $file",
            FileInputStream(file)
        )
    ).use {
        SerializationUtils.deserialize(it) as T
    }
}

val timestamp = ThreadLocal<Long>()
fun printlnStart(s: String) {
    timestamp.set(System.currentTimeMillis())
    println(s)
}

fun printlnEnd(s: String) {
    println(s + " Took ${(System.currentTimeMillis() - timestamp.get()) / 1000} s.")
    println("")
}

fun showPlot(array: Array<DoubleArray>) {
    val plot = LinePlot.of(array)
    plot.canvas().window().defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
}

fun subSet(
    X: Matrix,
    y: IntArray,
    nrows: Int
): Pair<Matrix, IntArray> {
    return if (nrows == -1) {
        X to y
    } else {
        X.submatrix(0, 0, nrows - 1, X.ncols() - 1) to y.copyOf(nrows)
    }
}

fun rotate90(x: Array<DoubleArray>, ncols: Int): Array<DoubleArray> {
    for (rIdx in x.indices) {
        val list = mutableListOf<DoubleArray>()
        val row = x[rIdx]
        for (i in row.indices step ncols) {
            list.add(row.copyOfRange(i, i + ncols))
        }
        x[rIdx] = Matrix(list.apply { reverse() }.toTypedArray()).transpose().toArray().flatMap { it.toList() }.toDoubleArray()
    }
    return x
}
