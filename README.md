# SandTube-Ai

Proyek fine-tuning adapter **LoRA** berbahasa Indonesia untuk model dasar [`izrai4/SandTube_Chat`](https://huggingface.co/izrai4/SandTube_Chat). Nama proyek dan adapter ini SandTube-Ai; ID dasar tetap dicantumkan agar adapter dapat dimuat dengan PEFT.

## Model dan ukuran

Bobot adapter `output/SandTube-Ai-LoRA/adapter_model.safetensors` berukuran **9.034.304 byte**, sama dengan adapter sebelumnya. Rank tetap 8, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`; jumlah parameter trainable 2.252.800. Adapter tidak di-merge ke bobot model dasar.

Model dasar diperlukan saat inference dan tidak disertakan di repo. Kartu Hugging Face model dasar saat ini belum menyebutkan lisensi. Periksa ketentuan upstream sebelum penggunaan atau redistribusi.

## Dataset dan reproduksi

Skrip `scripts/download_hf_datasets.py` mengunduh dataset publik yang dirujuk proyek: `cahya/instructions_indonesian`, `LorthGyu/indonesian-instructions`, dan `INTISARI/intisari-indonesian-chat-v5`. `scripts/prepare_sandtube_ai_data.py` membuat split yang sudah difilter. Run metadata tersedia di `data/huggingface/`.

File berisi contoh mentah dan split berisi teks dari dataset Hugging Face **tidak disertakan di repo**; lisensi redistribusinya belum diverifikasi. Jalankan downloader untuk mengambilnya dari sumber, lalu jalankan filter untuk menyiapkan data. Filter konservatif memilih 22 pasangan berbahasa Indonesia berisiko rendah; 19 dipakai untuk training dan 3 untuk validasi. Korpus training juga berisi entri dari `scripts/kbbi_lexicon.py` dan contoh perilaku dari `scripts/behavior_examples.py`. Jumlah keseluruhan: 209 baris train dan 13 validasi.

```bash
pip install -r requirements.txt
python scripts/download_hf_datasets.py
python scripts/prepare_sandtube_ai_data.py
MAX_STEPS=80 OUTPUT_DIR=output/SandTube-Ai-LoRA python scripts/train_sandtube_ai.py
```

Training yang tercatat menggunakan LoRA pada CPU selama 80 langkah. Loss train 1,7114; loss validasi 1,7668. Label prompt dimask agar loss hanya dihitung pada jawaban assistant. Skrip memberlakukan batas keras ukuran adapter 9.034.304 byte.

## Menjalankan chat

```bash
ADAPTER_PATH=output/SandTube-Ai-LoRA python scripts/chat_sandtube.py
```

Lima uji singkat tersimpan di `evaluation_sandtube_ai.json`. Benchmark yang lebih terstruktur tersedia di `evaluation_sandtube_ai_v2.json` dan dapat dijalankan dengan:

```bash
MODEL_NAME=izrai4/SandTube_Chat \
ADAPTER_PATH=output/SandTube-Ai-LoRA \
python scripts/evaluate_sandtube_ai.py
```

Hasil benchmark ditulis ke `evaluation_results.json`. Evaluator memberi skor per kategori untuk honesty, matematika, missing context, identitas, klarifikasi, dan pengetahuan dasar. Model belum boleh dipakai untuk keputusan faktual penting sebelum lulus gerbang evaluasi.

Rancangan eksperimen dan hasil uji terbaru ada di `MODEL_TEST_AND_TRAINING_PLAN.md`.
