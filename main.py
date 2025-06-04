import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# --- Carregamento e Preparação dos Dados ---
nome_arquivo_excel = 'trabalho_estatistica.xlsx'

try:
    df = pd.read_excel(nome_arquivo_excel)
    print(f"Arquivo '{nome_arquivo_excel}' carregado com sucesso.")
except FileNotFoundError:
    print(f"ARQUIVO NÃO ENCONTRADO: '{nome_arquivo_excel}'. Verifique o nome e o caminho do arquivo.")
    exit()

# Conversão de colunas com vírgula decimal para numérico (float)
cols_com_virgula_decimal = ['price_y', 'price_x']
for col in cols_com_virgula_decimal:
    if col in df.columns:
        if df[col].dtype == 'object': # Só tenta converter se for string/object
            print(f"\nConvertendo coluna '{col}' (que tem vírgula como decimal)...")
            # Substitui vírgula por ponto e converte para float
            # O errors='coerce' transformará em NaN o que não puder ser convertido
            df[col] = df[col].astype(str).str.replace('.', '', regex=False) # Remove separador de milhar se houver (ex: 1.234,56)
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float, errors='coerce')
            print(f"Coluna '{col}' convertida para tipo numérico.")
            # Verificar quantos NaNs foram criados, se houver
            if df[col].isnull().any():
                print(f"Atenção: Foram gerados valores ausentes (NaN) na coluna '{col}' durante a conversão. Verifique os dados originais.")
        elif pd.api.types.is_numeric_dtype(df[col]):
             print(f"Coluna '{col}' já é numérica. Nenhuma conversão de vírgula decimal necessária.")
        else:
            print(f"Atenção: Coluna '{col}' não é do tipo 'object' nem numérica. Tipo atual: {df[col].dtype}. Verifique os dados.")


    else:
        print(f"Aviso: Coluna '{col}' esperada para conversão decimal não encontrada no DataFrame.")

# Conversão de colunas de data
cols_data = ['created_at', 'updated_at']
for col_data in cols_data:
    if col_data in df.columns:
        # Tenta converter para datetime, inferindo o formato.
        # Para formatos mistos como 'dd/mm/yyyy' e 'yyyy-mm-dd', o pandas tenta adivinhar.
        # Se houver erro, pode ser necessário especificar o formato ou usar dayfirst=True
        df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
    else:
        print(f"Aviso: Coluna de data '{col_data}' não encontrada no DataFrame.")

# --- Análises Estatísticas ---
print("\n--- Estatísticas Descritivas Detalhadas ---")

df['price_y'] = df['price_y'].dropna() # Remove NaNs para os cálculos individuais
df['quantity'] = df['quantity'].dropna()
df['price_x'] = df['price_x'].dropna()
df['product_id'] = df['product_id'].dropna()

# Média
print('Média de faturamento')
media = df['price_x'].mean()
print(f"Média: {media:.2f}")

# Moda
print("Moda do produto mais vendido")
moda = df['product_id'].mode()
if not moda.empty:
    print(f"Moda: {', '.join(map(lambda x: f'{x:.2f}', moda.tolist()))}")
else:
    print("Moda: Não foi possível calcular.")

# Mediana
print("Mediana do preço dos produtos")
mediana = df['price_y'].median()
print(f"Mediana: {mediana:.2f}")

# Quartis (Q1, Q2, Q3)
q1 = df['price_y'].quantile(0.25)
q2 = df['price_y'].quantile(0.50) # Mediana
q3 = df['price_y'].quantile(0.75)
print(f"Quartis: Q1={q1:.2f}, Q2(Mediana)={q2:.2f}, Q3={q3:.2f}")
iqr = q3 - q1
print(f"Amplitude Interquartil (IQR): {iqr:.2f}")

# Percentis (exemplo: Percentil 10 e Percentil 90)
p10 = df['quantity'].quantile(0.10)
p90 = df['quantity'].quantile(0.90)
print(f"Percentil 10: {p10:.2f}")
print(f"Percentil 90: {p90:.2f}")

# Desvio Padrão
desvio_padrao = df['quantity'].std()
print(f"Desvio Padrão: {desvio_padrao:.2f}")

# Variância
variancia = df['quantity'].var()
print(f"Variância: {variancia:.2f}")

# # Teste de Normalidade (Shapiro-Wilk)
# if len(col_data) > 2 and len(col_data) <= 5000: # Shapiro-Wilk é bom para n <= 5000
#     try:
#         stat_shapiro, p_valor_shapiro = stats.shapiro(col_data)
#         print(f"Teste de Normalidade (Shapiro-Wilk): Estatística={stat_shapiro:.3f}, p-valor={p_valor_shapiro:.3f}")
#         if p_valor_shapiro > 0.05:
#             print("  -> Sugestão (Shapiro-Wilk): Os dados PARECEM ser normalmente distribuídos (não rejeitar H0).")
#         else:
#             print("  -> Sugestão (Shapiro-Wilk): Os dados NÃO PARECEM ser normalmente distribuídos (rejeitar H0).")
#     except Exception as e_shapiro:
#             print(f"Erro no teste de Shapiro-Wilk para {col}: {e_shapiro}")

# elif len(col_data) > 5000:
#     # Para N > 5000, Kolmogorov-Smirnov com correção de Lilliefors (usando statsmodels) é uma alternativa
#     try:
#         ks_stat, ks_p_value = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()), N=len(col_data))
#         print(f"Teste de Normalidade (Kolmogorov-Smirnov): Estatística={ks_stat:.3f}, p-valor={ks_p_value:.3f}")
#         if ks_p_value > 0.05:
#             print("  -> Sugestão (K-S): Os dados PARECEM ser normalmente distribuídos (não rejeitar H0).")
#         else:
#             print("  -> Sugestão (K-S): Os dados NÃO PARECEM ser normalmente distribuídos (rejeitar H0).")
#     except Exception as e_ks:
#         print(f"Erro no teste de Kolmogorov-Smirnov para {col}: {e_ks}")
# else:
#     print("Teste de Normalidade: Não realizado (dados insuficientes após remover NaNs).")

# # Histograma para visualização da distribuição
# plt.figure(figsize=(10, 5))
# sns.histplot(col_data, kde=True, bins=30)
# plt.title(f'Histograma e Curva de Densidade para {col}')
# plt.xlabel(col)
# plt.ylabel('Frequência / Densidade')
# plt.grid(axis='y', alpha=0.75)
# plt.show()


# print("\n--- Correlação e Regressão ---")
# numeric_df_for_corr = df[colunas_numericas_analise].copy() # Usa apenas as colunas selecionadas e válidas
# numeric_df_for_corr.dropna(inplace=True) # Remove linhas com NaN para correlação e regressão

# if numeric_df_for_corr.shape[0] > 1 and numeric_df_for_corr.shape[1] >= 2:
# # Matriz de Correlação (Pearson)
# matriz_correlacao = numeric_df_for_corr.corr(method='pearson')
# print("\nMatriz de Correlação (Pearson):")
# print(matriz_correlacao)

# # Mapa de calor da correlação
# plt.figure(figsize=(8, 6))
# sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Mapa de Calor da Matriz de Correlação')
# plt.show()

# # Regressão Linear Simples
# # Vamos usar 'price_x' (preço total) como dependente e 'quantity' (quantidade) como independente.
# # Outra opção seria 'price_x' ~ 'price_y'
# col_dependente = 'price_x'
# col_independente = 'quantity'

# if col_dependente in numeric_df_for_corr.columns and col_independente in numeric_df_for_corr.columns:
#     print(f"\nRegressão Linear Simples: {col_dependente} ~ {col_independente}")
#     try:
#         # Nomes seguros para a fórmula (caso tenham caracteres especiais, embora os seus não tenham)
#         safe_col_dependente = f'Q("{col_dependente}")'
#         safe_col_independente = f'Q("{col_independente}")'

#         formula = f'{safe_col_dependente} ~ {safe_col_independente}'
#         modelo = smf.ols(formula=formula, data=numeric_df_for_corr).fit()
#         print(modelo.summary())

#         # Gráfico de dispersão com a linha de regressão
#         plt.figure(figsize=(10, 6))
#         sns.regplot(x=col_independente, y=col_dependente, data=numeric_df_for_corr, ci=95, line_kws={'color':'red'}, scatter_kws={'alpha':0.5})
#         plt.title(f'Regressão Linear: {col_dependente} vs {col_independente}')
#         plt.xlabel(col_independente)
#         plt.ylabel(col_dependente)
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.show()

#         # Estatística Preditiva (Exemplo simples de previsão)
#         # Suponha que você queira prever 'price_x' para uma nova 'quantity'
#         if not numeric_df_for_corr[col_independente].empty:
#             novo_valor_X = numeric_df_for_corr[col_independente].median() # Usando a mediana como exemplo
#             df_para_previsao = pd.DataFrame({col_independente: [novo_valor_X]})
#             previsao_Y = modelo.predict(df_para_previsao)
#             print(f"Exemplo de Previsão de '{col_dependente}' para '{col_independente}' = {novo_valor_X:.2f}: {previsao_Y[0]:.2f}")
#         else:
#             print("Não foi possível gerar exemplo de previsão (coluna independente vazia).")


#     except Exception as e_reg:
#         print(f"Erro ao executar a regressão: {e_reg}")
# else:
#     print(f"Não foi possível realizar a regressão: colunas '{col_dependente}' ou '{col_independente}' não encontradas ou não são válidas após limpeza.")
# else:
# print("Dados insuficientes para calcular a matriz de correlação ou realizar regressão (verifique NaNs ou número de colunas/linhas).")


# print("\n--- Probabilidade e Distribuições (Exemplos) ---")

# # Distribuição Binomial (Exemplo Genérico)
# n_binomial, p_binomial, k_binomial = 10, 0.3, 6 # tentativas, prob. sucesso, num. sucessos
# prob_k_sucessos = stats.binom.pmf(k_binomial, n_binomial, p_binomial)
# print(f"\nDistribuição Binomial (n={n_binomial}, p={p_binomial}):")
# print(f"Prob. de exatamente {k_binomial} sucessos: {prob_k_sucessos:.4f} ({prob_k_sucessos*100:.2f}%)")
# # Gráfico da PMF
# x_binom = np.arange(0, n_binomial + 1)
# pmf_binom = stats.binom.pmf(x_binom, n_binomial, p_binomial)
# plt.figure(figsize=(8,4))
# plt.bar(x_binom, pmf_binom, label=f'PMF (n={n_binomial}, p={p_binomial})')
# plt.title('Distribuição Binomial - Função Massa de Probabilidade (PMF)')
# plt.xlabel('Número de Sucessos (k)')
# plt.ylabel('Probabilidade')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()


# # Distribuição Normal (Usando 'price_x' como exemplo, se disponível e válido)
# if 'price_x' in numeric_df_for_corr.columns and not numeric_df_for_corr['price_x'].empty:
# col_normal_exemplo = 'price_x'
# media_normal = numeric_df_for_corr[col_normal_exemplo].mean()
# std_normal = numeric_df_for_corr[col_normal_exemplo].std()

# if std_normal > 0: # Evita erro se todos os valores forem iguais
#     print(f"\nDistribuição Normal (exemplo com '{col_normal_exemplo}', média={media_normal:.2f}, dp={std_normal:.2f}):")
#     x_val_norm = media_normal + std_normal # Um valor exemplo
#     cdf_norm = stats.norm.cdf(x_val_norm, loc=media_normal, scale=std_normal)
#     pdf_norm = stats.norm.pdf(x_val_norm, loc=media_normal, scale=std_normal)
#     print(f"Prob. de valor <= {x_val_norm:.2f} (CDF): {cdf_norm:.4f}")
#     print(f"Densidade de prob. em {x_val_norm:.2f} (PDF): {pdf_norm:.4f}")

#     # Gráfico da PDF com dados
#     plt.figure(figsize=(10,5))
#     sns.histplot(numeric_df_for_corr[col_normal_exemplo], kde=False, stat="density", label='Histograma dos Dados (Densidade)', bins=30)
#     x_axis_norm = np.linspace(min(numeric_df_for_corr[col_normal_exemplo]), max(numeric_df_for_corr[col_normal_exemplo]), 100)
#     plt.plot(x_axis_norm, stats.norm.pdf(x_axis_norm, media_normal, std_normal), 'r-', lw=2, label='PDF Normal Teórica')
#     plt.title(f'Distribuição Normal Teórica vs Histograma de {col_normal_exemplo}')
#     plt.xlabel(col_normal_exemplo)
#     plt.ylabel('Densidade')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()
# else:
#     print(f"Não foi possível usar '{col_normal_exemplo}' para exemplo de Dist. Normal (desvio padrão é zero ou dados insuficientes).")
# else:
# print("\nExemplo de Distribuição Normal não gerado (coluna 'price_x' não disponível/válida).")


# # Distribuição Uniforme (Exemplo Genérico)
# a_unif, b_unif = 0, 20 # Limites da distribuição uniforme
# print(f"\nDistribuição Uniforme (entre {a_unif} e {b_unif}):")
# x_val_unif = (a_unif + b_unif) / 2
# cdf_unif = stats.uniform.cdf(x_val_unif, loc=a_unif, scale=(b_unif - a_unif))
# pdf_unif = stats.uniform.pdf(x_val_unif, loc=a_unif, scale=(b_unif - a_unif))
# print(f"Prob. de valor <= {x_val_unif} (CDF): {cdf_unif:.4f}")
# print(f"Densidade de prob. em {x_val_unif} (PDF): {pdf_unif:.4f}")
# # Gráfico da PDF
# x_axis_unif = np.linspace(a_unif - 5, b_unif + 5, 200)
# pdf_values_unif = stats.uniform.pdf(x_axis_unif, loc=a_unif, scale=(b_unif-a_unif))
# plt.figure(figsize=(8,4))
# plt.plot(x_axis_unif, pdf_values_unif, label=f'PDF Uniforme ({a_unif},{b_unif})')
# plt.title('Distribuição Uniforme - Função Densidade de Probabilidade (PDF)')
# plt.xlabel('Valores')
# plt.ylabel('Densidade')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

print("\n--- Fim das Análises ---")