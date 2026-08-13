# MVP — Gestão Geoespacial de Modelos DAI

Aplicativo Streamlit para organizar modelos de avaliação imobiliária, visualizar a incidência territorial dos trabalhos técnicos e apoiar a priorização de atualizações.

O projeto parte de duas fontes:

1. banco SQLite com trabalhos, imóveis, coordenadas e modelos utilizados;
2. catálogos seguros extraídos localmente dos arquivos `.dai`.

> O aplicativo web **não abre arquivos `.dai`**. Esses arquivos usam `joblib/pickle`, podem depender de versões específicas do ambiente Python e só devem ser carregados localmente quando sua origem for confiável.

## Funcionalidades

- mapa de densidade dos trabalhos técnicos;
- gráficos de barras por modelo, família e ano;
- catálogo com período da amostra, tamanho, outliers e R² ajustado;
- mapa comparando trabalhos e pontos da amostra do modelo;
- distância de cada trabalho ao dado de treinamento mais próximo;
- correção automática de coordenadas históricas invertidas;
- triagem de modelos por demanda, recência, suporte espacial e presença no catálogo;
- modo de demonstração com dados totalmente sintéticos;
- envio transitório do SQLite e dos CSVs pela interface.

## Estrutura

```text
.
├── app.py
├── data/demo/                 # somente dados sintéticos
├── scripts/
│   ├── extract_dai.py         # extrator local para .dai confiáveis
│   └── generate_demo_data.py
├── src/
│   ├── charts.py
│   ├── config.py
│   ├── data.py
│   └── metrics.py
├── tests/
├── requirements.txt
└── requirements-extractor.txt
```

## Executar localmente

Recomenda-se Python 3.11 ou 3.12.

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

No Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

O modo inicial usa os arquivos de `data/demo`.

## Usar os dados reais sem publicá-los

### 1. Preparar o extrator

```bash
pip install -r requirements-extractor.txt
```

### 2. Extrair os `.dai` localmente

```bash
python scripts/extract_dai.py \
  --input-dir "CAMINHO/PARA/MODELOS" \
  --output-dir data/private \
  --trust-source
```

O argumento `--trust-source` é uma confirmação explícita de que os pickles são internos e confiáveis.

Saídas:

- `data/private/catalogo_modelos.csv`;
- `data/private/amostras_modelos.csv.gz`;
- `data/private/erros_extracao.csv`.

Por padrão, o extrator não exporta nomes, matrículas, documentos, endereços nem valores individuais. Exporta somente metadados do modelo e coordenadas necessárias para o diagnóstico espacial.

### 3. Copiar o banco

Copie o banco para:

```text
data/private/trabalhos_tecnicos.sqlite3
```

Essa pasta está bloqueada pelo `.gitignore`. Quando os arquivos existem, a opção **Arquivos locais protegidos** aparece na barra lateral.

Também é possível escolher **Enviar arquivos nesta sessão**, sem gravá-los no repositório.

## Publicar no GitHub

```bash
git init
git add .
git commit -m "MVP de gestão geoespacial de modelos DAI"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/gestao-modelos-dai.git
git push -u origin main
```

Antes do `git add`, confirme:

```bash
git status
```

Não devem aparecer arquivos `.dai`, bancos reais nem `data/private`.

## Publicar no Streamlit Community Cloud

1. envie este repositório ao GitHub;
2. acesse o Streamlit Community Cloud;
3. escolha **Create app**;
4. selecione o repositório e a branch `main`;
5. informe `app.py` como arquivo principal;
6. faça o deploy.

Para uma aplicação pública, mantenha apenas o modo de demonstração. O envio de arquivos reais pela interface deve ser usado apenas em sessão controlada e consciente de que o endereço da aplicação é público.

## Escore de triagem

O MVP calcula uma fila operacional:

| Componente | Peso |
|---|---:|
| Demanda nos dois anos mais recentes do banco | 35% |
| Antiguidade do fim da amostra | 25% |
| Distância P90 dos trabalhos à amostra | 25% |
| Ausência no catálogo atual | 15% |

O escore não declara que um modelo é inválido. Ele serve para localizar situações que merecem auditoria técnica.

## Limitações do MVP

- distância espacial não mede extrapolação multivariada;
- o nome histórico ainda não substitui um vínculo por hash/versionamento formal;
- a diferença entre valor estimado e adotado não está disponível no banco atual;
- cobertura autorizada e vigência precisam de cadastro institucional;
- o mapa usa serviços públicos de tiles e requer internet no navegador.

## Próximos passos recomendados

1. criar tabelas de `modelo_releases`, aliases, vigência e substituição;
2. registrar SHA-256 em cada novo trabalho;
3. acrescentar estimativa, intervalo, valor adotado e motivo do ajuste;
4. migrar a camada institucional para PostgreSQL/PostGIS;
5. autenticar a aplicação antes de disponibilizar dados reais.

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

## Licença

Código disponibilizado sob a licença MIT. A licença não se estende aos dados, modelos ou documentos técnicos da Prefeitura.

