# Segurança e privacidade

## Arquivos `.dai`

Os arquivos são serializações `joblib/pickle`. A desserialização pode executar código. Execute `scripts/extract_dai.py` apenas sobre arquivos internos e confiáveis, em máquina controlada.

O aplicativo Streamlit não recebe nem abre `.dai`.

## Dados que não devem ser publicados

- modelos `.dai` originais;
- banco real de trabalhos técnicos;
- nomes e documentos de transmitentes ou adquirentes;
- endereços ou inscrições imobiliárias não destinados à divulgação;
- catálogos derivados que preservem registros pessoais;
- credenciais e segredos de infraestrutura.

## Streamlit Community Cloud

Considere pública qualquer aplicação implantada sem autenticação institucional. O modo demonstrativo usa dados sintéticos. Para dados reais, prefira infraestrutura interna autenticada.

## Relato de vulnerabilidades

Não abra uma issue pública contendo dados reais. Use o canal institucional responsável pelo repositório.

