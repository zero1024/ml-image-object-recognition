package poa.ml.image.objects.recognition.labeler

import poa.ml.image.objects.recognition.clone
import java.awt.Color
import java.awt.Graphics
import java.awt.Point
import java.awt.Rectangle
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.event.MouseMotionAdapter
import java.awt.image.BufferedImage
import javax.swing.ImageIcon
import javax.swing.JLabel
import kotlin.math.abs
import kotlin.math.min


class DrawableJLabel(image: BufferedImage, minSize: Int) : JLabel(ImageIcon(image.clone())) {

    private var p1: Point = Point(0, 0)
    private var p2: Point = Point(0, 0)
    private var drawing = false

    init {
        addMouseListener(object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                drawing = true
                p1 = e.point
                p2 = p1
                repaint()
            }

            override fun mouseReleased(e: MouseEvent) {
                drawing = false
                p2 = e.point
                repaint()
            }

        })
        addMouseMotionListener(object : MouseMotionAdapter() {
            override fun mouseDragged(e: MouseEvent) {
                if (abs(e.point.x - p1.x) > minSize && abs(e.point.y - p1.y) > minSize) {
                    p2 = e.point
                    repaint()
                }
            }
        })
    }

    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        g.color = Color.WHITE
        val r = getRectangle()
        g.drawRect(r.x, r.y, r.width, r.height)
    }

    fun getRectangle(): Rectangle {
        val leftX = min(p1.x, p2.x)
        val leftY = min(p1.y, p2.y)
        val width = abs(p1.x - p2.x)
        val height = abs(p1.y - p2.y)
        return Rectangle(leftX, leftY, width, height)
    }

}