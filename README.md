# s_koha_work

## 素敵なコマンド集

### 特定のファイル形式のファイル数を求める

```bash
find <dir> -name "*.wav" | wc -l
```

### 各ディレクトリの容量を確認（depth: 1）

```bash
du -h -d 1 .
```