# cvtColor numerical data

The two little-endian payloads are generated from OpenCV contrib `2f5682ba` with the
OpenCV `40738fb16ceddb5fb3fea747585f7ce6abb0605b` softfloat implementation.
The script also generates `src/imgproc/cvtcolor_codes.inc`. Use a scratch output
directory and compare the generated files before replacing shipped data.

```sh
python tools/generate_cvtcolor_tables.py /path/to/opencv_contrib /path/to/opencv \
  /path/to/pinned-opencv-install /path/to/output
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| lab-luv-reverse.bin | 1227776 | c152c0855596a3b7ac56e0c046492f641dc0d71cda4ea45a37d4349c49bf108c |
| luv-forward.bin | 1724976 | a669b7a5f0fc27857b20256c5efb171a0d6a3f48a62cb5e4abd0d4e3164f7d67 |
| cvtcolor_codes.inc | 9912 | a72d3747d670e6bd5672ff7b2822e75193fa9f2e12b3b3c1a7698b022c30ade2 |
