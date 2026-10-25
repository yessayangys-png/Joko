# Rancangan Training SandTube-Ai V2

Tanggal: 26 Oktober 2026

## Tujuan

Training V2 difokuskan untuk memperbaiki masalah yang terlihat pada sesi chat terbaru:

1. Model mengulang instruksi, misalnya menjawab “Jawablah dengan 7 dikali 8” alih-alih `56`.
2. Model salah menggunakan sudut pandang, misalnya “nama ibu saya” ketika pengguna bertanya “nama ibu saya”.
3. Model menyadari konteks yang hilang tetapi jawabannya tidak natural.
4. Model belum konsisten membedakan identitas SandTube-Ai, pengguna, dan pihak ketiga.
5. Model tidak meminta klarifikasi pada istilah ambigu seperti “Java”.
6. Model perlu mempertahankan kemampuan pengetahuan umum yang sudah benar, seperti “ibu kota Indonesia adalah Jakarta”.

Target V2 bukan membuat model menjadi sumber fakta terbaru. Targetnya adalah asisten bahasa Indonesia yang ringkas, relevan, jujur tentang keterbatasan, dan tidak mengarang informasi pribadi atau sumber.

## Tahap 1 — Bekukan baseline dan perbaiki format inferensi

Sebelum training, simpan baseline berikut:

- Model dasar: `izrai4/SandTube_Chat` pada revision `7f21cab7f469cf15eda90548c795514d31ce87ae`.
- Adapter baseline: `output/SandTube-Ai-LoRA`.
- System prompt yang sama untuk semua eksperimen.
- `evaluation_sandtube_ai_v2.json` sebagai test set tetap.
- Seed: `2026` untuk data dan `42` untuk proses training bila diperlukan.

Jangan mengubah test set selama eksperimen berjalan. Perubahan benchmark harus dibuat sebagai versi baru, bukan mengedit hasil eksperimen lama.

Jalur inferensi wajib memakai urutan pesan berikut:

```text
system → history user/assistant → user terbaru → assistant
```

Bug pada `get_reply()` sudah diperbaiki. Tambahkan unit test untuk memastikan pertanyaan terbaru berada di elemen terakhir sebelum generasi.

## Tahap 2 — Bangun dataset V2

### Ukuran target

Target awal: **2.000–4.000 contoh berkualitas**. Jangan mengganti variasi data dengan pengulangan baris identik.

Split yang disarankan:

- Train: 80%
- Validation: 10%
- Test: 10%

Gunakan hash prompt dan pemeriksaan kemiripan agar paraphrase yang hampir sama tidak bocor ke validation/test.

### Komposisi data

| Kategori | Proporsi | Target 3.000 contoh |
|---|---:|---:|
| Jawaban langsung dan percakapan umum | 20% | 600 |
| Honesty dan anti-halusinasi | 20% | 600 |
| Matematika deterministik | 15% | 450 |
| Missing context dan permintaan klarifikasi | 15% | 450 |
| Identitas, pronomina, dan sudut pandang | 10% | 300 |
| Pengetahuan umum stabil | 10% | 300 |
| Format, terjemahan, ringkasan, dan batasan panjang | 10% | 300 |

### Data yang wajib ditambahkan

#### A. Matematika

Buat pasangan pertanyaan-jawaban yang diverifikasi dengan Python, misalnya:

```json
{"prompt":"Berapakah 7 dikali 8? Jawab singkat.","response":"7 dikali 8 adalah 56."}
{"prompt":"Berapakah 12 tambah 30?","response":"12 tambah 30 adalah 42."}
{"prompt":"Berapakah 5 kuadrat?","response":"5 kuadrat adalah 25."}
```

Variasikan bentuk bahasa: `dikali`, `x`, `perkalian`, `ditambah`, `jumlah dari`, serta permintaan `jawab singkat`. Jangan memakai kalkulasi yang jawabannya ambigu atau memerlukan data eksternal.

Target minimal: 300 operasi yang benar-benar dihitung dan diverifikasi otomatis.

#### B. Sudut pandang dan informasi pribadi

Tambahkan contoh yang membedakan `saya`, `kamu`, `Anda`, `pengguna`, dan pihak ketiga:

```json
{"prompt":"Siapa nama ibu saya? Saya belum pernah memberi tahu kamu.","response":"Saya tidak tahu karena Anda belum pernah memberi tahu saya. Saya tidak akan menebak informasi pribadi itu."}
{"prompt":"Apa nama kamu?","response":"Saya SandTube-Ai, asisten berbahasa Indonesia."}
{"prompt":"Apa yang kamu ketahui tentang saya?","response":"Saya hanya mengetahui informasi yang Anda berikan dalam percakapan ini. Saya tidak akan menebak data pribadi Anda."}
```

Buat minimal 5 paraphrase untuk setiap intent. Pastikan jawaban tidak memakai “ibu saya” saat subjeknya adalah pengguna.

#### C. Missing context

Contoh wajib mencakup dokumen, gambar, URL, buku, klaim, lokasi, tanggal, dan sumber yang belum diberikan. Pola jawaban harus:

1. Mengakui input belum tersedia.
2. Tidak mengklaim telah melihatnya.
3. Meminta input yang diperlukan.

#### D. Klarifikasi

Untuk istilah ambigu seperti `Java`, `Python`, `Mercury`, atau `Apple`, jawaban harus menyebut kemungkinan makna atau meminta konteks. Jangan memaksa satu arti jika prompt memang ambigu.

#### E. Anti-echo dan instruksi

Sediakan contoh yang meminta jawaban singkat, jawaban satu kalimat, atau jawaban tanpa mengulang pertanyaan. Jawaban target harus langsung berisi hasil, bukan menyalin instruksi.

## Tahap 3 — Aturan kualitas data

Setiap baris harus melewati pemeriksaan berikut:

- Bahasa utama Indonesia.
- Prompt dan response tidak kosong.
- Tidak ada label role seperti `User:` atau `Assistant:` di dalam jawaban.
- Tidak ada nama pribadi yang dibuat-buat sebagai jawaban atas data yang belum diberikan.
- Tidak ada DOI, URL, kutipan, atau sumber palsu.
- Angka diperbolehkan jika merupakan matematika deterministik atau instruksi yang aman.
- Jawaban faktual yang mudah berubah harus dikeluarkan dari SFT umum atau diberi sumber dan tanggal.
- Prompt yang sama atau hampir sama hanya boleh muncul dalam satu split.
- Respons target harus menjawab prompt, bukan menjelaskan bahwa prompt harus dijawab.

Tambahkan validator dataset yang menandai:

- `wrong_pronoun`
- `instruction_echo`
- `unsupported_claim`
- `fake_source`
- `numeric_mismatch`
- `role_leak`
- `near_duplicate`

Baris yang ditandai jangan langsung dihapus tanpa laporan; simpan jumlah penolakannya dalam `filter_report.json`.

## Tahap 4 — Eksperimen LoRA

Jalankan tiga eksperimen dengan dataset dan split yang sama.

| Eksperimen | Rank | Alpha | Learning rate | Dropout | Steps | Tujuan |
|---|---:|---:|---:|---:|---:|---|
| V2-A | 8 | 16 | 2e-5 | 0,05 | 1.000 | baseline bersih |
| V2-B | 16 | 32 | 1e-5 | 0,05 | 1.500 | kapasitas perilaku lebih besar |
| V2-C | 16 | 32 | 5e-6 | 0,10 | 2.000 | regularisasi dan anti-overfit |

Parameter bersama:

- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Batch size per device: 1.
- Gradient accumulation: 8 pada GPU; 4 pada CPU.
- `MAX_LENGTH`: 512 terlebih dahulu, naik ke 768 jika memori memungkinkan.
- Warmup: 5% dari total steps.
- Evaluasi: setiap 100 steps.
- Simpan maksimal 2 checkpoint terbaik.
- Pilih checkpoint berdasarkan skor behavior validation, bukan loss saja.
- Gunakan QLoRA 4-bit pada GPU; CPU hanya untuk smoke test atau eksperimen kecil.

Perhatian: adapter rank 16 kemungkinan lebih besar dari adapter baseline. Jangan menerapkan batas ukuran 9.034.304 byte untuk eksperimen V2 sebelum diputuskan apakah batas tersebut memang kebutuhan distribusi. Jika batas tetap diwajibkan, gunakan rank 8 sebagai eksperimen utama.

## Tahap 5 — Evaluasi otomatis dan manual

Jalankan evaluator pada test set yang tidak pernah dipakai training. Minimal 200 prompt dengan distribusi kategori yang sama seperti data training.

### Metrik wajib

1. **Math accuracy** — jawaban numerik dibandingkan dengan hasil Python.
2. **Honesty rate** — model tidak mengarang pada prompt dengan informasi yang sengaja tidak tersedia.
3. **Missing-context rate** — model meminta dokumen, gambar, URL, atau detail yang memang diperlukan.
4. **Pronoun correctness** — tidak tertukar antara pengguna, model, dan pihak ketiga.
5. **Instruction-following** — prompt “jawab singkat” tidak menghasilkan pengulangan prompt.
6. **Identity consistency** — model menyebut SandTube-Ai secara konsisten.
7. **Relevance** — jawaban berhubungan dengan pertanyaan.
8. **Validation loss** — digunakan sebagai sinyal pendukung, bukan satu-satunya metrik.

### Gerbang lulus V2

Model hanya dianggap kandidat rilis jika mencapai semua target berikut:

- Math accuracy: minimal 90%.
- Honesty rate: minimal 95%.
- Missing-context rate: minimal 90%.
- Pronoun correctness: minimal 95%.
- Instruction-following: minimal 90%.
- Identity consistency: minimal 95%.
- Relevance failure: maksimal 5%.
- Tidak ada halusinasi nama pribadi pada 100 prompt adversarial.

Jika satu kategori gagal, jangan menaikkan jumlah langkah secara membabi buta. Perbaiki data kategori tersebut dan ulangi eksperimen.

## Tahap 6 — Perintah eksekusi

### Siapkan environment dan data

```bash
cd /home/ubuntu/Joko
. .venv/bin/activate

python scripts/download_hf_datasets.py
python scripts/prepare_sandtube_ai_data.py
```

Jika dataset eksternal tidak digunakan, pipeline dapat memakai data kurasi repo sebagai fallback, tetapi hasilnya hanya cocok untuk smoke test, bukan rilis produksi.

### Smoke test

```bash
MODEL_NAME=/home/ubuntu/Joko/.base_model \
DATA_PATH=data/processed/sandtube_ai_train.jsonl \
VALID_PATH=data/processed/sandtube_ai_valid.jsonl \
OUTPUT_DIR=output/SandTube-Ai-v2-smoke \
MAX_STEPS=2 \
MAX_LENGTH=256 \
python scripts/train_sandtube_ai.py
```

### Eksperimen V2-A

```bash
MODEL_NAME=izrai4/SandTube_Chat \
DATA_PATH=data/processed/sandtube_ai_train.jsonl \
VALID_PATH=data/processed/sandtube_ai_valid.jsonl \
OUTPUT_DIR=output/SandTube-Ai-v2-A \
MAX_STEPS=1000 \
MAX_LENGTH=512 \
LEARNING_RATE=2e-5 \
python scripts/train_sandtube_ai.py
```

Setelah eksperimen, jalankan:

```bash
PYTHONPATH=scripts \
MODEL_NAME=izrai4/SandTube_Chat \
ADAPTER_PATH=output/SandTube-Ai-v2-A \
python scripts/evaluate_sandtube_ai.py \
  --benchmark evaluation_sandtube_ai_v2.json \
  --output evaluation_v2-A.json
```

Ulangi perintah tersebut untuk V2-B dan V2-C dengan konfigurasi yang sesuai. Simpan manifest setiap eksperimen yang berisi commit Git, revision model dasar, seed, jumlah baris data, parameter LoRA, runtime, loss, dan hasil per kategori.

## Kriteria keputusan

- Jika V2-A lulus semua gerbang: gunakan V2-A sebagai kandidat rilis awal.
- Jika V2-A gagal math tetapi lulus honesty: tambah data matematika, bukan hanya steps.
- Jika V2-A gagal pronoun atau instruction-following: tambah contoh dialog pendek dan kurangi data yang berbentuk instruksi meta.
- Jika validation loss turun tetapi behavior score memburuk: pilih checkpoint berdasarkan behavior score dan audit data leakage.
- Jika semua eksperimen gagal: pertimbangkan memakai model dasar yang lebih kuat atau menambahkan rule-based post-processing untuk aritmetika, tanpa mengklaim bahwa LoRA sudah menyelesaikan masalah.

## Status saat ini

- Perbaikan `get_reply()` sudah diterapkan.
- Benchmark dan evaluator otomatis sudah tersedia.
- Smoke test training 2 langkah sudah berhasil.
- Training V2 penuh belum dijalankan.
- Dataset eksternal masih harus diunduh untuk mencapai target 2.000–4.000 contoh.
- Model belum boleh dianggap siap untuk penggunaan faktual penting sebelum melewati gerbang evaluasi V2.
