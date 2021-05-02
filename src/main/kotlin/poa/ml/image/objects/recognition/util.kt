package poa.ml.image.objects.recognition

import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Toolkit
import java.awt.image.BufferedImage
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.WindowConstants


fun showImage(image: BufferedImage, lambda: (JFrame) -> Unit = {}) {
    val frame = JFrame()
    val dim: Dimension = Toolkit.getDefaultToolkit().screenSize
    frame.contentPane.layout = FlowLayout()
    frame.contentPane.add(JLabel(ImageIcon(image)))
    frame.defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
    lambda(frame)
    frame.setLocation(dim.width / 2 - frame.size.width / 2, dim.height / 2 - frame.size.height / 2)
    frame.pack()
    frame.isVisible = true
}