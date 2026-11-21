package com.eew.ai

import android.app.Activity
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.random.Random

class MainActivity : Activity() {
    private enum class Lang { ZH, EN }

    private val REPEAT_PENALTY = 2.0f
    private val CJK_PUNCT = "，。！？、；：“”‘’（）《》…"
    private val EN_PUNCT = ".,!?;:'\"()-"

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var scroll: ScrollView
    private lateinit var chatBox: LinearLayout
    private lateinit var input: EditText
    private lateinit var sendBtn: Button
    private lateinit var clearBtn: Button

    private var history = JSONArray()
    private var ready = false

    // 字表按"码点"切分，与 Python 的字符数量保持一致
    private var chars: List<String> = listOf()
    private var charIndex: HashMap<String, Int> = HashMap()
    private var ctx = 0
    private var dim = 0
    private var heads = 0
    private var layers = 0
    private var ff = 0
    private var vocab = 0

    private var emb = FloatArray(0)
    private var pos = FloatArray(0)
    private var lnfW = FloatArray(0)
    private var lnfB = FloatArray(0)
    private var headW = FloatArray(0)
    private var headB = FloatArray(0)
    private lateinit var blocks: Array<Block>

    private class Block(
        val ln1W: FloatArray, val ln1B: FloatArray,
        val inW: FloatArray, val inB: FloatArray,
        val outW: FloatArray, val outB: FloatArray,
        val ln2W: FloatArray, val ln2B: FloatArray,
        val ff1W: FloatArray, val ff1B: FloatArray,
        val ff2W: FloatArray, val ff2B: FloatArray
    )

    private val historyFile by lazy { File(filesDir, "chat_history.json") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        chatBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 16, 16, 16)
        }
        scroll = ScrollView(this).apply { addView(chatBox) }

        input = EditText(this).apply { hint = "输入开头文字（中文或英文）" }
        sendBtn = Button(this).apply { text = "发送" }
        clearBtn = Button(this).apply { text = "清空对话" }

        val inputRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(sendBtn)
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(clearBtn)
            addView(inputRow)
        }
        setContentView(root)

        loadHistory()
        setBusy(true)
        addBubble("ai", "正在加载模型…", save = false)

        Thread {
            try {
                loadModel()
                ui.post {
                    ready = true
                    setBusy(false)
                    addBubble("ai", "模型已加载，输入文字即可开始。", save = false)
                }
            } catch (e: Exception) {
                ui.post { addBubble("ai", "加载失败：$e", save = false) }
            }
        }.start()

        sendBtn.setOnClickListener {
            val prompt = input.text.toString().trim()
            if (prompt.isEmpty() || !ready) return@setOnClickListener
            input.setText("")
            setBusy(true)
            addBubble("user", prompt, save = true)
            val pending = addBubble("ai", "…", save = false)
            Thread {
                val reply = try {
                    generate(prompt, 120)
                } catch (e: Exception) {
                    "出错：$e"
                }
                ui.post {
                    pending.text = reply
                    appendHistory("ai", reply)
                    setBusy(false)
                }
            }.start()
        }

        clearBtn.setOnClickListener {
            history = JSONArray()
            saveHistory()
            chatBox.removeAllViews()
            addBubble("ai", "对话已清空。", save = false)
        }
    }

    // ---------- 对话记录（永久保存在手机本地） ----------

    private fun loadHistory() {
        history = try {
            if (historyFile.exists()) JSONArray(historyFile.readText(Charsets.UTF_8)) else JSONArray()
        } catch (e: Exception) {
            JSONArray()
        }
        for (i in 0 until history.length()) {
            val item = history.getJSONObject(i)
            addBubble(item.getString("role"), item.getString("text"), save = false)
        }
    }

    private fun saveHistory() {
        historyFile.writeText(history.toString(), Charsets.UTF_8)
    }

    private fun appendHistory(role: String, text: String) {
        history.put(JSONObject().put("role", role).put("text", text))
        saveHistory()
    }

    // ---------- 气泡界面 ----------

    private fun addBubble(role: String, text: String, save: Boolean): TextView {
        val isUser = role == "user"
        val tv = TextView(this).apply {
            this.text = text
            textSize = 16f
            setTextColor(Color.BLACK)
            setPadding(28, 18, 28, 18)
            maxWidth = resources.displayMetrics.widthPixels * 3 / 4
            background = GradientDrawable().apply {
                cornerRadius = 24f
                setColor(if (isUser) Color.parseColor("#95EC69") else Color.WHITE)
            }
        }
        val lp = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = if (isUser) Gravity.END else Gravity.START
            setMargins(0, 12, 0, 12)
        }
        chatBox.addView(tv, lp)
        if (save) appendHistory(role, text)
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        return tv
    }

    private fun setBusy(busy: Boolean) {
        sendBtn.isEnabled = !busy && ready
        input.isEnabled = !busy && ready
    }

    // ---------- 字符工具（按码点切分） ----------

    private fun splitCodePoints(s: String): List<String> =
        s.codePoints().toArray().map { String(Character.toChars(it)) }

    // ---------- 模型加载 ----------

    private fun loadModel() {
        val text = assets.open("model.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
        val json = JSONObject(text)

        chars = splitCodePoints(json.getString("chars"))
        charIndex = HashMap()
        chars.forEachIndexed { i, c -> charIndex[c] = i }
        vocab = chars.size

        ctx = json.getInt("ctx")
        dim = json.getInt("dim")
        heads = json.getInt("heads")
        layers = json.getInt("layers")
        ff = json.getInt("ff")

        emb = flat(json.getJSONArray("emb"))
        pos = flat(json.getJSONArray("pos"))
        lnfW = flat(json.getJSONArray("ln_f_w"))
        lnfB = flat(json.getJSONArray("ln_f_b"))
        headW = flat(json.getJSONArray("head_w"))
        headB = flat(json.getJSONArray("head_b"))

        // 检查字表与参数是否对得上，对不上就给出明确提示，而不是运行时崩溃
        if (headB.size != vocab) {
            throw IllegalStateException("字表长度 $vocab 与模型输出 ${headB.size} 不一致")
        }
        if (emb.size != vocab * dim) {
            throw IllegalStateException("嵌入大小与字表不一致")
        }

        blocks = Array(layers) { i ->
            val b = json.getJSONObject("l$i")
            Block(
                flat(b.getJSONArray("ln1_w")), flat(b.getJSONArray("ln1_b")),
                flat(b.getJSONArray("in_w")), flat(b.getJSONArray("in_b")),
                flat(b.getJSONArray("out_w")), flat(b.getJSONArray("out_b")),
                flat(b.getJSONArray("ln2_w")), flat(b.getJSONArray("ln2_b")),
                flat(b.getJSONArray("ff1_w")), flat(b.getJSONArray("ff1_b")),
                flat(b.getJSONArray("ff2_w")), flat(b.getJSONArray("ff2_b"))
            )
        }
    }

    private fun flat(a: JSONArray): FloatArray =
        FloatArray(a.length()) { a.getDouble(it).toFloat() }

    // ---------- 语言判断 ----------

    private fun isCjkCp(cp: Int): Boolean = cp in 0x4e00..0x9fff

    private fun detectLang(s: String): Lang {
        if (s.codePoints().anyMatch { isCjkCp(it) }) return Lang.ZH
        if (s.codePoints().anyMatch { it in 'a'.code..'z'.code || it in 'A'.code..'Z'.code }) return Lang.EN
        return Lang.ZH
    }

    private fun allowedFor(lang: Lang, c: String): Boolean {
        val cp = c.codePointAt(0)
        if (c == "\n" || cp in '0'.code..'9'.code) return true
        return when (lang) {
            Lang.ZH -> isCjkCp(cp) || c == " " || CJK_PUNCT.contains(c)
            Lang.EN -> (cp in 'a'.code..'z'.code) || (cp in 'A'.code..'Z'.code) ||
                    c == " " || EN_PUNCT.contains(c)
        }
    }

    // ---------- 数学工具 ----------

    // x: (n, inD)；W 按 PyTorch 的存法 (outD, inD) 行优先，即 W[j*inD + i]；结果 (n, outD)
    private fun linear(x: FloatArray, n: Int, inD: Int, W: FloatArray, b: FloatArray, outD: Int): FloatArray {
        val out = FloatArray(n * outD)
        for (t in 0 until n) {
            for (j in 0 until outD) {
                var s = b[j]
                val base = j * inD
                for (i in 0 until inD) s += x[t * inD + i] * W[base + i]
                out[t * outD + j] = s
            }
        }
        return out
    }

    private fun layerNorm(x: FloatArray, n: Int, d: Int, w: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(n * d)
        for (t in 0 until n) {
            var mean = 0f
            for (i in 0 until d) mean += x[t * d + i]
            mean /= d
            var v = 0f
            for (i in 0 until d) {
                val c = x[t * d + i] - mean
                v += c * c
            }
            val inv = 1f / sqrt(v / d + 1e-5f)
            for (i in 0 until d) out[t * d + i] = (x[t * d + i] - mean) * inv * w[i] + b[i]
        }
        return out
    }

    // GELU 的近似实现，与 PyTorch 的精确 GELU 差异很小
    private fun gelu(v: Float): Float {
        return 0.5f * v * (1f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)))
    }

    // 多头自注意力（因果掩码），输入输出 (T, D)
    private fun attention(x: FloatArray, T: Int, blk: Block): FloatArray {
        val D = dim
        val hd = D / heads
        val qkv = linear(x, T, D, blk.inW, blk.inB, 3 * D)
        val out = FloatArray(T * D)
        val scale = 1f / sqrt(hd.toFloat())

        for (h in 0 until heads) {
            for (t in 0 until T) {
                val scores = FloatArray(t + 1)
                var mx = -1e30f
                for (s in 0..t) {
                    var dot = 0f
                    for (d in 0 until hd) {
                        val qv = qkv[t * 3 * D + h * hd + d]
                        val kv = qkv[s * 3 * D + D + h * hd + d]
                        dot += qv * kv
                    }
                    scores[s] = dot * scale
                    if (scores[s] > mx) mx = scores[s]
                }
                var sum = 0f
                for (s in 0..t) {
                    scores[s] = exp(scores[s] - mx)
                    sum += scores[s]
                }
                for (s in 0..t) {
                    val a = scores[s] / sum
                    for (d in 0 until hd) {
                        out[t * D + h * hd + d] += a * qkv[s * 3 * D + 2 * D + h * hd + d]
                    }
                }
            }
        }
        return linear(out, T, D, blk.outW, blk.outB, D)
    }

    // ---------- 前向推理（只取最后一个位置的输出） ----------

    private fun nextLogits(window: IntArray): FloatArray {
        val T = ctx
        val D = dim

        var h = FloatArray(T * D)
        for (t in 0 until T) {
            val id = window[t]
            for (d in 0 until D) h[t * D + d] = emb[id * D + d] + pos[t * D + d]
        }

        for (blk in blocks) {
            val a = layerNorm(h, T, D, blk.ln1W, blk.ln1B)
            val att = attention(a, T, blk)
            for (i in h.indices) h[i] += att[i]

            val b = layerNorm(h, T, D, blk.ln2W, blk.ln2B)
            val f1 = linear(b, T, D, blk.ff1W, blk.ff1B, ff)
            for (i in f1.indices) f1[i] = gelu(f1[i])
            val f2 = linear(f1, T, ff, blk.ff2W, blk.ff2B, D)
            for (i in h.indices) h[i] += f2[i]
        }

        val last = FloatArray(D) { h[(T - 1) * D + it] }
        val lastN = layerNorm(last, 1, D, lnfW, lnfB)
        return linear(lastN, 1, D, headW, headB, vocab)
    }

    // ---------- 生成 ----------

    private fun generate(prompt: String, n: Int): String {
        val lang = detectLang(prompt)
        val allowed = BooleanArray(vocab) { allowedFor(lang, chars[it]) }

        val seq = ArrayList<Int>()
        for (c in splitCodePoints(prompt)) charIndex[c]?.let { seq.add(it) }
        if (seq.isEmpty()) return "输入的文字不在模型词表中"

        val gen = ArrayList<Int>()
        repeat(n) {
            val window = IntArray(ctx) { i ->
                val idx = seq.size - ctx + i
                if (idx >= 0) seq[idx] else seq[0]
            }
            val logits = nextLogits(window)

            val recent = gen.takeLast(30).toSet()
            for (v in 0 until vocab) {
                if (!allowed[v]) {
                    logits[v] = -1e9f
                } else if (v in recent) {
                    logits[v] -= REPEAT_PENALTY
                }
            }

            val id = sample(logits, 0.8f)
            seq.add(id)
            gen.add(id)
        }
        return gen.joinToString("") { chars[it] }
    }

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
