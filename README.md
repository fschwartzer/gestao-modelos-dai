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
- fila de intervenção P0–P3 separando governança, completude e escore auxiliar;
- exportação CSV versionada da fila filtrada, com data de referência e cobertura das evidências;
- consolidação automática das revisões alfabéticas, mantendo somente a mais recente;
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
3. Envie o banco de trabalhos com extensão `.sqlite`, `.sqlite3` ou `.db` e/ou selecione um ou
   mais modelos `.DAI`.
4. Se houver `.DAI`, confirme que os modelos são internos e confiáveis.

Os arquivos não são incorporados ao repositório. O catálogo, o hash SHA-256 e as coordenadas da
amostra são extraídos em memória. Falhas em um `.DAI` são registradas sem descartar os demais.

As fontes são independentes e as páginas são habilitadas somente quando seus insumos existem:

| Análise | SQLite | `.DAI`/catálogo | Amostra espacial |
|---|:---:|:---:|:---:|
| Visão geral dos trabalhos | Obrigatório | — | — |
| Catálogo de modelos | — | Obrigatório | — |
| Cobertura espacial | Obrigatório | Obrigatório | Obrigatório |
| Triagem integrada | Obrigatório | Obrigatório | Obrigatório |
| Metodologia | — | — | — |

Assim, é possível trabalhar somente com SQLite ou somente com `.DAI`; as análises incompatíveis
ficam fora da navegação e a barra lateral informa quais fontes estão disponíveis.

### Revisões dos modelos

Quando modelos compartilham a mesma base numérica, a letra final representa a revisão. A maior
revisão alfabética é considerada a atual: por exemplo, `MOD_V_TER_Z1_006J` substitui
`MOD_V_TER_Z1_006I`.

A revisão mais recente é identificada pela união dos nomes presentes no SQLite, no catálogo e nas
amostras. Os usos históricos são consolidados nessa revisão para que cobertura e triagem não contem
versões da mesma linhagem como modelos independentes. Catálogos e amostras de revisões antigas são
descartados, e nunca atribuídos à revisão nova. Assim, se o `.DAI` atual estiver ausente, o aplicativo
não usa silenciosamente as datas, métricas ou coordenadas de treinamento da versão anterior.

### Arquivos locais protegidos

Também é possível manter os dados fora do Git em:

```text
data/private/
├── trabalhos_tecnicos.sqlite3
└── modelos/
    ├── MODELO_1.dai
    └── MODELO_2.dai
```

A pasta está bloqueada pelo `.gitignore`. A opção **Arquivos locais protegidos** aparece quando
qualquer uma das fontes existe. Arquivos `.DAI` podem ficar diretamente em `data/private/` ou em
`data/private/modelos/`; seu processamento é opcional e exige confirmação de origem confiável.

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

No mapa, todas as revisões mais recentes com situação **Vigente** ou **Alerta** são selecionadas por
padrão. Modelos classificados como **Não utilizar** permanecem disponíveis para seleção manual e
auditoria, mas não entram na visualização inicial.

## Fila de intervenção e escore auxiliar

A decisão é organizada em camadas independentes:

| Camada | Resultado |
|---|---|
| Governança temporal | Vigente, Alerta ou Não utilizar |
| Completude | Avaliável, Incompleta ou Sem evidência |
| Prioridade | P0, P1, P2 ou P3 |
| Escore auxiliar | Desempate dentro da mesma classe |

Além do escore, aplica-se uma regra temporal obrigatória sobre a data do dado mais contemporâneo:

| Idade do dado mais contemporâneo | Situação |
|---|---|
| Até 6 meses | Vigente |
| Acima de 6 e até 12 meses | Alerta |
| Acima de 12 meses | Não utilizar |
| Data ausente | Não utilizar |

Os limites usam mês-calendário: exatamente 6 meses ainda é vigente e exatamente 12 meses permanece
em alerta. A situação temporal é uma regra mandatória e não recebe peso no escore.

As classes de ação são:

| Classe | Interpretação operacional |
|---|---|
| P0 | Não utilizar com demanda recente: suspender e atualizar/substituir imediatamente |
| P1 | Atualização prioritária ou decisão entre atualizar e aposentar |
| P2 | Revisão programada, alerta sem demanda ou evidências incompletas |
| P3 | Monitoramento periódico |

A fila, os gráficos, o mapa, a tabela e o CSV obedecem aos filtros de anos, tipos de trabalho e
famílias. Uma seleção parcial gera uma visão analítica parcial; para auditar o portfólio completo,
é necessário selecionar todas as opções. A demanda procura uma data completa do trabalho e usa
janela móvel de 12 meses. Quando o SQLite contém somente o ano, o cálculo usa o ano civil atual e
o anterior como aproximação, deixando o método visível.

O escore auxiliar usa uma escala fixa de demanda: 20 trabalhos na janela atingem impacto máximo,
valor configurável em `src/config.py`. Ele não é normalizado pelo modelo mais demandado do arquivo
atual. Sua estrutura de evidências é:

| Componente | Peso planejado |
|---|---:|
| Impacto operacional | 40% |
| Risco de desempenho em teste | 35% |
| Suporte espacial/extrapolação | 15% |
| Risco operacional e completude | 10% |

O componente de desempenho permanece ausente até existir uma base de teste identificada. Evidência
ausente não recebe penalidade técnica nem é redistribuída entre os demais componentes. O aplicativo
mostra a cobertura percentual e rotula o escore como provisório. R², MSE ou resíduos de ajuste não
são usados como substitutos de COD, PRD, mediana das razões, regressividade e estabilidade fora da
amostra.

O suporte espacial deixa de usar um limite universal de 5 km. Para cada modelo, compara-se a distância
P90 dos trabalhos recentes à amostra com a distância P90 entre vizinhos da própria amostra. A parcela
de trabalhos fora da envoltória convexa também participa do diagnóstico. O risco de distância cresce
linearmente de zero, na razão 1×, até o máximo, na razão 3×; depois é combinado com a parcela fora da
envoltória. A unidade permanece quilômetro, em WGS84, com distância haversine. Esse diagnóstico é
pós-modelagem e não cria features de treino. Resultados com menos de 10 trabalhos georreferenciados
são marcados como exploratórios e não alteram automaticamente a classe de prioridade.

## Limitações do MVP

- distância espacial não mede extrapolação multivariada;
- a envoltória convexa pode superestimar suporte em amostras descontínuas;
- a revisão inferida pelo nome não substitui um vínculo institucional por hash/versionamento formal;
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

