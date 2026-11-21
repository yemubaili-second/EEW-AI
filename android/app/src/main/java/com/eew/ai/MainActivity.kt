package com.eew.ai

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.tanh
import kotlin.random.Random

class MainActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var output: TextView
    private lateinit var input: EditText
    private lateinit var button: Button

    private var chars: List<Char> = listOf()
    private var charIndex: HashMap<Char, Int> = HashMap()
    private var ctx = 0
    private var dim = 0
    private var hidden = 0
    private var vocab = 0
    private var emb = FloatArray(0)
    private var w1 = FloatArray(0)
    private var b1 = FloatArray(0)
    private var w2 = FloatArray(0)
    private var b2 = FloatArray(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        output = TextView(this).apply { text = "正在加载模型…" }
        input = EditText(this).apply { hint = "输入开头文字" }
        button = Button(this).apply { text = "生成"; isEnabled = false }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(30, 30, 30, 30)
        }
        root.addView(
            ScrollView(this).apply { addView(output) },
            LinearLayout.LayoutParams(-1, 0, 1f)
        )
        root.addView(input)
        root.addView(button)
        setContentView(root)

        Thread {
            try {
                loadModel()
                ui.post {
                    output.text = "模型已加载，输入文字后点击生成"
                    button.isEnabled = true
                }
            } catch (e: Exception) {
                ui.post { output.text = "加载失败：$e" }
            }
        }.start()

        button.setOnClickListener {
            val prompt = input.text.toString()
            button.isEnabled = false
            output.text = "生成中…"
            Thread {
                val result = generate(prompt, 120)
                ui.post {
                    output.text = prompt + result
                    button.isEnabled = true
                }
            }.start()
        }
    }

    private fun loadModel() {
        val text = assets.open("model.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
        val json = JSONObject(text)
        chars = json.getString("chars").toList()
        charIndex = HashMap()
        chars.forEachIndexed { i, c -> charIndex[c] = i }
        vocab = chars.size
        ctx = json.getInt("ctx")
        dim = json.getInt("dim")
        hidden = json.getInt("hidden")
        emb = flat(json.getJSONArray("emb"))
        w1 = flat(json.getJSONArray("w1"))
        b1 = flat(json.getJSONArray("b1"))
        w2 = flat(json.getJSONArray("w2"))
        b2 = flat(json.getJSONArray("b2"))
    }

    private fun flat(a: JSONArray): FloatArray =
        FloatArray(a.length()) { a.getDouble(it).toFloat() }

    // 前向计算：输入前 ctx 个字的编号，输出每个字的得分
    private fun nextLogits(window: IntArray): FloatArray {
        val xdim = ctx * dim
        val x = FloatArray(xdim)
        for (i in 0 until ctx) {
            val id = window[i]
            for (d in 0 until dim) x[i * dim + d] = emb[id * dim + d]
        }
        val h = FloatArray(hidden)
        for (j in 0 until hidden) {
            var s = b1[j]
            for (k in 0 until xdim) s += x[k] * w1[k * hidden + j]
            h[j] = tanh(s)
        }
        val logits = FloatArray(vocab)
        for (v in 0 until vocab) {
            var s = b2[v]
            for (j in 0 until hidden) s += h[j] * w2[j * vocab + v]
            logits[v] = s
        }
        return logits
    }

    private fun generate(prompt: String, n: Int): String {
        val seq = ArrayList<Int>()
        for (c in prompt) charIndex[c]?.let { seq.add(it) }
        if (seq.isEmpty()) seq.add(Random.nextInt(vocab))

        val out = StringBuilder()
        repeat(n) {
            val window = IntArray(ctx) { i ->
                val idx = seq.size - ctx + i
                if (idx >= 0) seq[idx] else seq[0]
            }
            val id = sample(nextLogits(window), 0.8f)
            seq.add(id)
            out.append(chars[id])
        }
        return out.toString()
    }

    // 按概率随机抽取，温度越低越保守
    private fun sample(logits: FloatArray, temp: Float): Int {
        var max = logits[0]
        for (v in logits) if (v > max) max = v
        val p = FloatArray(logits.size) { exp((logits[it] - max) / temp) }
        var r = Random.nextFloat() * p.sum()
        for (i in p.indices) {
            r -= p[i]
            if (r <= 0f) return i
        }
        return p.size - 1
    }
}
