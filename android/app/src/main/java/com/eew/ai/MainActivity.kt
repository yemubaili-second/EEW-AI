package com.example.llama

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private val llm = LLamaAndroid.instance()
    private lateinit var history: TextView
    private lateinit var input: EditText
    private lateinit var sendBtn: Button
    private var corpus: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        history = TextView(this).apply { text = "正在加载模型…" }
        input = EditText(this).apply { hint = "输入你的问题" }
        sendBtn = Button(this).apply { text = "发送"; isEnabled = false }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(30, 30, 30, 30)
        }
        root.addView(ScrollView(this).apply { addView(history) },
            LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(input)
        root.addView(sendBtn)
        setContentView(root)

        CoroutineScope(Dispatchers.IO).launch {
            val modelFile = File(filesDir, "model.gguf")
            if (!modelFile.exists()) {
                assets.open("model.gguf").use { i ->
                    modelFile.outputStream().use { o -> i.copyTo(o) }
                }
            }
            corpus = assets.open("corpus.txt").bufferedReader(Charsets.UTF_8).readText()
            llm.load(modelFile.absolutePath)
            withContext(Dispatchers.Main) {
                history.text = "模型已加载，可以提问了。"
                sendBtn.isEnabled = true
            }
        }

        sendBtn.setOnClickListener {
            val q = input.text.toString().trim()
            if (q.isEmpty()) return@setOnClickListener
            input.setText("")
            sendBtn.isEnabled = false
            history.append("\n\n你：$q\nAI：")

            val hits = corpus.split("\n")
                .filter { line -> line.length > 4 && q.any { c -> line.contains(c) } }
                .take(5)
                .joinToString("\n")
            val prompt = if (hits.isEmpty()) q
                else "请参考以下资料回答问题：\n$hits\n\n问题：$q"

            CoroutineScope(Dispatchers.IO).launch {
                llm.send(prompt).collect { piece ->
                    withContext(Dispatchers.Main) { history.append(piece) }
                }
                withContext(Dispatchers.Main) { sendBtn.isEnabled = true }
            }
        }
    }
}
