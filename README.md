# MVP — Gestão Geoespacial de Modelos DAI

Aplicativo Streamlit para organizar modelos de avaliação imobiliária, visualizar a incidência territorial dos trabalhos técnicos e apoiar a priorização de atualizações.

O projeto parte de duas fontes:

1. banco SQLite com trabalhos, imóveis, coordenadas e modelos utilizados;
2. um ou mais arquivos `.DAI` com os modelos e suas amostras espaciais.

> Arquivos `.DAI` usam `joblib/pickle` e podem executar código durante a leitura. O aplicativo exige confirmação explícita antes de abri-los, mas isso não substitui uma sandbox. Use o envio direto somente com arquivos internos e confiáveis.

## Funcionalidades

- mapa de densidade dos trabalhos técnicos;
- gráficos de barras por modelo, família e ano;
- catálogo com período da amostra, tamanho, outliers e R² ajustado;
- mapa multicamadas sobrepondo trabalhos e envoltórias empíricas dos modelos;
- classificação dos trabalhos dentro/fora da envoltória convexa de cada amostra;
- distância de cada trabalho ao dado de treinamento mais próximo;
- correção automática de coordenadas históricas invertidas;
- triagem de modelos por demanda, recência, suporte espacial e presença no catálogo;
- modo de demonstração com dados totalmente sintéticos;
- envio transitório de múltiplos `.DAI` e um banco SQLite pela interface;
- alternativa compatível com catálogos CSV pré-extraídos.

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
│   ├── dai.py
│   ├── data.py
│   ├── metrics.py
│   └── spatial.py
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

## Usar os dados reais em uma sessão local

1. Execute o aplicativo localmente.
2. Na barra lateral, escolha **Enviar arquivos nesta sessão**.
3. Envie o banco de trabalhos com extensão `.sqlite`, `.sqlite3` ou `.db`.
4. Selecione um ou mais modelos `.DAI`.
5. Confirme que os modelos são internos e confiáveis.

Os arquivos não são incorporados ao repositório. O catálogo, o hash SHA-256 e as coordenadas da
amostra são extraídos em memória. Falhas em um `.DAI` são registradas sem descartar os demais.

### Arquivos locais protegidos

Também é possível manter os dados fora do Git em:

```text
data/private/
├── trabalhos_tecnicos.sqlite3
└── modelos/
    ├── MODELO_1.dai
    └── MODELO_2.dai
```

A pasta está bloqueada pelo `.gitignore`. Quando o banco existe, a opção **Arquivos locais
protegidos** aparece na barra lateral. Arquivos `.DAI` podem ficar diretamente em `data/private/`
ou em `data/private/modelos/`.

### Alternativa: extrair catálogos antes da sessão

Para uma implantação que não deve desserializar `.DAI`, execute o extrator em ambiente controlado:

```bash
pip install -r requirements-extractor.txt
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

Para uma aplicação pública, mantenha apenas o modo de demonstração e defina a variável de ambiente
`ALLOW_DAI_UPLOADS=false`. Isso remove o uploader `.DAI`; os CSVs pré-extraídos continuam disponíveis
como alternativa. O envio de dados reais deve ocorrer apenas em implantação autenticada e controlada.

## Alcance espacial no mapa

- CRS das coordenadas: WGS84 (`EPSG:4326`);
- unidade das distâncias: quilômetros, calculados por haversine;
- regra de alcance: envoltória convexa dos pontos válidos da amostra de cada modelo;
- mínimo: três pontos distintos e não colineares;
- custo: contenção ponto-polígono por modelo e busca haversine `O(trabalhos × amostra)`;
  a busca é executada em blocos de até um milhão de pares para limitar o uso de memória.

A envoltória é um diagnóstico empírico: pode atravessar lacunas sem observações e não representa
zona legal, vigência institucional ou ausência de extrapolação multivariada. A análise é
pós-modelagem e não cria features, portanto não mistura treino e teste. COD, PRD, mediana das razões
e regressividade ainda exigem valores observados e estimados em uma base de teste ou validação
espacial/temporal identificada.

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
- a envoltória convexa pode superestimar suporte em amostras descontínuas;
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

