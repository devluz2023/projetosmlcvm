import json
import os


def unificar_arquivos(
    extensoes=(
        ".py",
        ".tf",
        ".sql",
        ".ipynb",
        ".sh",
        ".yml",
        ".yaml",
        ".md",
        ".json",
        ".toml",
        ".java",
        ".xml",
        ".Dockerfile",
    ),
):
  # Define o diretório como a pasta raiz onde o script está localizado
  diretorio = os.path.dirname(os.path.abspath(__file__))
  nome_arquivo_saida = "codigo_unificado.txt"
  arquivo_saida = os.path.join(diretorio, nome_arquivo_saida)
  pastas_ignoradas = {"logs", "env", "venv"}

  print(f"Buscando arquivos na raiz do projeto: {diretorio}")

  contador = 0
  with open(arquivo_saida, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(diretorio):
      # Ignora pastas ocultas (começadas com '.') e pastas indesejadas
      dirs[:] = [
          d
          for d in dirs
          if not d.startswith(".") and d.lower() not in pastas_ignoradas
      ]

      for file in files:
        filepath = os.path.join(root, file)

        # Pula o próprio script e o arquivo de saída gerado
        if file == os.path.basename(__file__) or file == nome_arquivo_saida:
          continue

        if file.endswith(extensoes):
          rel_path = os.path.relpath(filepath, diretorio)

          print(f"Lendo: {rel_path}")
          contador += 1

          # Cabeçalho para separar cada arquivo no txt final
          outfile.write("=" * 80 + "\n")
          outfile.write(f"ARQUIVO: {rel_path}\n")
          outfile.write("=" * 80 + "\n\n")

          if file.endswith(".ipynb"):
            try:
              with open(filepath, "r", encoding="utf-8") as f:
                nb = json.load(f)
                for cell in nb.get("cells", []):
                  cell_type = cell.get("cell_type")
                  source = "".join(cell.get("source", []))
                  if cell_type == "code":
                    outfile.write("# [Célula de Código]\n")
                    outfile.write(source + "\n\n")
                  elif cell_type == "markdown":
                    outfile.write("# [Célula Markdown]\n")
                    outfile.write(source + "\n\n")
            except Exception as e:
              outfile.write(f"# Erro ao ler o notebook: {e}\n")
          else:
            try:
              with open(filepath, "r", encoding="utf-8") as f:
                outfile.write(f.read())
            except Exception as e:
              outfile.write(f"# Erro ao ler o arquivo: {e}\n")

          outfile.write("\n\n")

  print(
      f"\nConcluído! {contador} arquivos foram unidos em: {arquivo_saida}"
  )


if __name__ == "__main__":
  unificar_arquivos()