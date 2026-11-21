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

    // ---------- 模型 ----------

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

    private fun isCjk(c: Char): Boolean = c in '\u4e00'..'\u9fff'

    private fun isLatin(c: Char): Boolean = c in 'a'..'z' || c in 'A'..'Z' || c == ' '

    private fun detectLang(s: String): Lang {
        if (s.any { isCjk(it) }) return Lang.ZH
        if (s.any { it in 'a'..'z' || it in 'A'..'Z' }) return Lang.EN
        return Lang.ZH
    }

    private fun allowedFor(lang: Lang, c: Char): Boolean {
        if (c == '\n' || c in '0'..'9') return true
        return when (lang) {
            Lang.ZH -> isCjk(c) || c == ' ' || c in CJK_PUNCT
            Lang.EN -> isLatin(c) || c in EN_PUNCT
        }
    }

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
        val lang = detectLang(prompt)
        val allowed = BooleanArray(vocab) { allowedFor(lang, chars[it]) }

        val seq = ArrayList<Int>()
        for (c in prompt) charIndex[c]?.let { seq.add(it) }
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
        return gen.joinToString("") { chars[it].toString() }
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
