package com.eew.ai

import android.app.Activity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import kotlin.random.Random

class MainActivity : Activity() {
    private lateinit var model: JSONObject
    private lateinit var chars: JSONArray
    private lateinit var output: TextView
    private lateinit var input: EditText
    private lateinit var button: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        input = EditText(this).apply { hint = "输入开头文字" }
        button = Button(this).apply { text = "生成"; isEnabled = false }
        output = TextView(this).apply { text = "正在加载模型..." }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 40)
        }
        layout.addView(input)
        layout.addView(button)
        layout.addView(
            ScrollView(this).apply { addView(output) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        )
        setContentView(layout)

        Thread {
            val text = assets.open("model.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
            model = JSONObject(text)
            chars = model.getJSONArray("chars")
            runOnUiThread {
                output.text = "模型已加载，输入文字后点击生成"
                button.isEnabled = true
            }
        }.start()

        button.setOnClickListener {
            val prompt = input.text.toString()
            button.isEnabled = false
            output.text = "生成中..."
            Thread {
                val result = generate(prompt, 200)
                runOnUiThread {
                    output.text = result
                    button.isEnabled = true
                }
            }.start()
        }
    }

    private fun generate(prompt: String, n: Int): String {
        val sb = StringBuilder(prompt)
        val uni = model.getJSONArray("uni")
        val bi = model.getJSONObject("bi")
        val tri = model.getJSONObject("tri")
        repeat(n) {
            val len = sb.length
            val next = when {
                len >= 2 && tri.has(sb.substring(len - 2)) ->
                    pick(tri.getJSONArray(sb.substring(len - 2)))
                len >= 1 && bi.has(sb.substring(len - 1)) ->
                    pick(bi.getJSONArray(sb.substring(len - 1)))
                else -> pick(uni)
            }
            sb.append(chars.getString(next))
        }
        return sb.toString()
    }

    // 每项为 [字的编号, 出现次数]，按次数加权随机抽取
    private fun pick(arr: JSONArray): Int {
        var total = 0
        for (i in 0 until arr.length()) total += arr.getJSONArray(i).getInt(1)
        var r = Random.nextInt(total)
        for (i in 0 until arr.length()) {
            val pair = arr.getJSONArray(i)
            r -= pair.getInt(1)
            if (r < 0) return pair.getInt(0)
        }
        return arr.getJSONArray(0).getInt(0)
    }
}
