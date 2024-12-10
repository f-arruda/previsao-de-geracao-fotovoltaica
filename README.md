#  APLICAÇÃO DO MODELO ENCODER DECODER LSTM PARA PREVISÃO DE GERAÇÃO FOTOVOLTAICA
Este repositório contém os resultados do projeto desenvolvido durante o Mestrado no programa Interdisciplinar em Ciência e Tecnologia do Mar da UNIFESP, com apoio financeiro da FAPESP (processo 2022/10281-6). O objetivo principal foi prever a geração fotovoltaica utilizando técnicas de aprendizado de máquina, com um enfoque em Transfer Learning para diferentes localidades do Brasil.

## Descrição do Projeto
Foi implementado um modelo Encoder-Decoder LSTM para prever a geração fotovoltaica em horizontes de 1, 2 e 3 horas no futuro. Os dados utilizados para o treinamento foram coletados de um gerador fotovoltaico em operação no Instituto de Energia e Meio Ambiente da USP (IEE/USP), localizado na Região Metropolitana de São Paulo. Após o treinamento, o modelo foi adaptado para outras regiões do Brasil utilizando a metodologia de _Transfer Learning_.

## Conteúdo do Repositório
Este repositório inclui:
1. **Dados**
   - Conjuntos de dados utilizados para o treinamento dos modelos e para a aplicação de _Transfer Learning_.
2. **Scripts**
   - Scripts para montagem e validação da base de dados.
   - Rotinas para treinamento dos modelos e análise das previsões.
   - Implementação do _Transfer Learning_.
   
3. **Modelos Treinados**
   -Pesos e configurações dos modelos treinados no desenvolvimento do projeto.
 
4. **Publicações**
   - Artigos submetidos e aprovados:
      - Artigo CBENS 2024.
      - Artigo CBPE 2024.
   - Dissertação de Mestrado.

Projeto desenvolvido durante o Mestrado no programa Interdisciplinar em Ciência e Tecnologia do Mar, da UNIFESP, com auxilio financeiro da FAPESP (processo 2022/10281-6).

Durante o desenvolvimento do projeto foi utilizado um modelo Encoder-Decoder LSTM para prever a geração fotovoltaica, no horizonte de previsão de 1, 2 e 3 horas no futuro. Para isso foi utilizado um gerador fotovoltaico em operação no Instituto de Energia e Meio Ambiente da USP (IEE/USP), localizado na Região Metropolitana de São Paulo. Após treinamento, foi utilizado a metodologia de transfer learning para avaliar o modelo em outras localidades do Brasil.

O repositório conta com:

- Dados utilizados para treinamento dos modelos e aplicação do Transfer Learning;
- Scripts utilizados para montagem e validação da base de dados, treinamento dos modelos, analise das previsões e aplicação do Transfer Learning;
- Modelos treinados no desenvolvimento do projeto; e
- Trabalhos publicados no decorrer do projeto: artigo CBENS 2024, artigo CBPE 2024 e Dissertação de Mestrado.
