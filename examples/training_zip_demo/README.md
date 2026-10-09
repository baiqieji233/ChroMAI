# Training ZIP Demo

This directory is a minimal raw dataset accepted by `maichart preprocess
build` and by the WebUI upload control. The generated archive
`examples/maimai_training_demo.zip` contains the same layout.

```text
training_demo/
  song_001/
    maidata.txt
    track.wav
  song_002/
    maidata.txt
    track.wav
```

Each song directory needs one `maidata.txt` and one sibling audio file. The
chart text contains a Master difficulty and demonstrates taps, a chord, a
hold, a slide, a touch note, and a local BPM directive. The WAV files are
short synthetic tones intended only to verify the data path; they are not
musical training material.

The ZIP intentionally contains no radar/style fields. Those controls are
derived in memory from frame labels when training samples are loaded.

Example preparation:

```powershell
maichart preprocess build examples/training_zip_demo/raw_dataset `
  --cache-dir examples/training_zip_demo/cache `
  -o examples/training_zip_demo/training_manifest.json `
  --division 16 --sample-rate 22050
```

Install the optional audio dependencies before running this command:
`py -m pip install -e ".[audio,training]"`.
