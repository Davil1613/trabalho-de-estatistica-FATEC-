import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional

# --- FUNÇÕES DE CONFIGURAÇÃO E PREPARAÇÃO ---

def configurar_visualizacao():
    """Configura o estilo dos gráficos para melhor visualização."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 75
    plt.rcParams['font.size'] = 12

def carregar_dados(nome_arquivo: str) -> pd.DataFrame:
    """
    Carrega os dados do arquivo Excel.
    
    Args:
        nome_arquivo: Caminho para o arquivo Excel.
        
    Returns:
        DataFrame com os dados carregados.
    """
    try:
        df = pd.read_excel(nome_arquivo)
        print(f"Arquivo '{nome_arquivo}' carregado com sucesso. {df.shape[0]} linhas encontradas.")
        return df
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{nome_arquivo}' não encontrado. Verifique o nome e o caminho do arquivo.")
        exit()

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e prepara os dados para análise estatística, 
    filtrando para os 5 anos mais recentes.
    
    Args:
        df: DataFrame original.
        
    Returns:
        DataFrame limpo e preparado para análise.
    """
    # Renomeando colunas para facilitar o uso
    df = df.copy()
    df.rename(columns={'price_x': 'total_venda', 'price_y': 'valor_unitario'}, inplace=True)
    
    # Conversão de colunas numéricas
    cols_numericas = ['valor_unitario', 'total_venda']
    for col in cols_numericas:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Convertendo coluna '{col}' para tipo numérico...")
                df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df[col].isnull().sum() > 0:
                    print(f"Atenção: {df[col].isnull().sum()} valores não puderam ser convertidos na coluna '{col}'.")
    
    # Conversão de colunas de data
    cols_data = ['created_at', 'updated_at']
    for col_data in cols_data:
        if col_data in df.columns:
            df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
            
    # --- NOVO CÓDIGO ADICIONADO ---
    # Objetivo: Manter no DataFrame apenas os registros dos 5 anos mais recentes.
    print("\nFiltrando dados para os 5 anos mais recentes...")
    if 'created_at' in df.columns and not df['created_at'].dropna().empty:
        # 1. Encontra o ano mais recente presente nos dados
        ano_mais_recente = df['created_at'].max().year
        
        # 2. Calcula qual é o primeiro ano do nosso intervalo de 5 anos.
        #    (Ex: Se o ano mais recente for 2025, o intervalo será 2021, 2022, 2023, 2024, 2025)
        ano_inicio = ano_mais_recente - 4
        
        print(f"  • Ano mais recente encontrado: {ano_mais_recente}")
        print(f"  • Mantendo dados a partir do ano de {ano_inicio} em diante.")
        
        linhas_antes = df.shape[0]
        
        # 3. Filtra o DataFrame, mantendo apenas as linhas cujo ano em 'created_at' é maior ou igual ao ano de início.
        df = df[df['created_at'].dt.year >= ano_inicio].copy()
        
        print(f"  • Linhas antes do filtro de data: {linhas_antes}")
        print(f"  • Linhas após o filtro de data: {df.shape[0]}")
    else:
        print("  • ATENÇÃO: Coluna 'created_at' não encontrada ou vazia. O filtro por ano não foi aplicado.")
    # --- FIM DO NOVO CÓDIGO ---

    # Remove linhas onde as colunas essenciais são nulas (agora sobre o df já filtrado)
    df_limpo = df.dropna(subset=['total_venda', 'quantity', 'valor_unitario', 'product_id']).copy()
    print(f"\nApós limpeza de dados nulos, restam {df_limpo.shape[0]} linhas para análise.")
    
    return df_limpo

# --- FUNÇÕES DE CÁLCULO ESTATÍSTICO ---

def calcular_estatisticas_descritivas(df: pd.DataFrame, coluna: str) -> Dict[str, float]:
    """Calcula estatísticas descritivas básicas para uma coluna."""
    descritivas = df[coluna].describe()
    
    # Calculando estatísticas adicionais
    amplitude = df[coluna].max() - df[coluna].min()
    iqr = descritivas['75%'] - descritivas['25%']
    cv = (df[coluna].std() / df[coluna].mean()) * 100  # Coeficiente de variação
    
    resultados = {
        'media': descritivas['mean'],
        'mediana': descritivas['50%'],
        'desvio_padrao': descritivas['std'],
        'variancia': df[coluna].var(),
        'minimo': descritivas['min'],
        'maximo': descritivas['max'],
        'amplitude': amplitude,
        'q1': descritivas['25%'],
        'q3': descritivas['75%'],
        'iqr': iqr,
        'coef_variacao': cv
    }
    
    return resultados

def testar_normalidade(df: pd.DataFrame, coluna: str) -> Tuple[float, float, bool]:
    """Realiza teste de normalidade de Shapiro-Wilk."""
    # Para grandes conjuntos de dados, usar uma amostra
    amostra = df[coluna]
    if len(amostra) > 5000:
        print(f"Aviso: Amostrando 5000 registros para o teste de normalidade devido ao tamanho da base.")
        amostra = df[coluna].sample(5000, random_state=42)
    
    stat, p_valor = stats.shapiro(amostra)
    eh_normal = p_valor > 0.05
    
    return stat, p_valor, eh_normal

def realizar_teste_t(df: pd.DataFrame, grupo1: pd.Series, grupo2: pd.Series, nome_var: str) -> Dict[str, Any]:
    """Realiza teste t para comparação de médias entre dois grupos."""
    _, p_norm1, normal1 = testar_normalidade(pd.DataFrame({nome_var: grupo1}), nome_var)
    _, p_norm2, normal2 = testar_normalidade(pd.DataFrame({nome_var: grupo2}), nome_var)
    
    stat_levene, p_levene = stats.levene(grupo1, grupo2)
    variancias_iguais = p_levene > 0.05
    
    stat_t, p_valor = stats.ttest_ind(grupo1, grupo2, equal_var=variancias_iguais)
    
    return {
        'estatistica_t': stat_t,
        'p_valor': p_valor,
        'diferenca_significativa': p_valor < 0.05,
        'media_grupo1': grupo1.mean(),
        'media_grupo2': grupo2.mean(),
        'diferenca_medias': grupo1.mean() - grupo2.mean(),
        'variancias_iguais': variancias_iguais,
        'normal_grupo1': normal1,
        'normal_grupo2': normal2
    }

def calcular_correlacao(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """Calcula correlação entre duas variáveis."""
    coef = df[col1].corr(df[col2])
    
    if abs(coef) < 0.3: interpretacao = "fraca"
    elif abs(coef) < 0.7: interpretacao = "moderada"
    else: interpretacao = "forte"
        
    direcao = "positiva" if coef > 0 else "negativa"
    
    n = len(df)
    t_stat = coef * np.sqrt((n - 2) / (1 - coef**2))
    p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    
    return {
        'coeficiente': coef,
        'interpretacao': f"Correlação {direcao} {interpretacao}",
        'r_squared': coef**2,
        'p_valor': p_valor,
        'significativa': p_valor < 0.05
    }

def realizar_regressao(df: pd.DataFrame, var_x: str, var_y: str) -> Dict[str, Any]:
    """Realiza regressão linear simples entre duas variáveis."""
    Y = df[var_y]
    X = sm.add_constant(df[var_x])
    modelo = sm.OLS(Y, X).fit()
    
    intercepto = modelo.params.iloc[0]
    coef_angular = modelo.params.iloc[1]
    
    df_aux = df[[var_x, var_y]].copy()
    df_aux['valor_predito'] = intercepto + coef_angular * df_aux[var_x]
    df_aux['residuos'] = df_aux[var_y] - df_aux['valor_predito']
    
    return {
        'modelo': modelo,
        'formula': f"{var_y} = {intercepto:.4f} + {coef_angular:.4f} * {var_x}",
        'r_squared': modelo.rsquared,
        'r_squared_adj': modelo.rsquared_adj,
        'p_valor': modelo.f_pvalue,
        'significativo': modelo.f_pvalue < 0.05,
        'dados_auxiliares': df_aux,
        'intercepto': intercepto,
        'coef_angular': coef_angular
    }

def calcular_probabilidades(df: pd.DataFrame, coluna: str) -> Dict[str, float]:
    """Calcula probabilidades empíricas baseadas nos dados."""
    media = df[coluna].mean()
    mediana = df[coluna].median()
    
    prob_maior_media = (df[coluna] > media).mean()
    prob_maior_mediana = (df[coluna] > mediana).mean()
    prob_entre_q1_q3 = ((df[coluna] >= df[coluna].quantile(0.25)) & 
                          (df[coluna] <= df[coluna].quantile(0.75))).mean()
    
    return {
        'prob_maior_media': prob_maior_media,
        'prob_maior_mediana': prob_maior_mediana,
        'prob_entre_q1_q3': prob_entre_q1_q3
    }

def calcular_dist_binomial(p_sucesso: float, n: int, k: int) -> float:
    """Calcula a probabilidade binomial."""
    return stats.binom.pmf(k=k, n=n, p=p_sucesso)

def calcular_dist_normal(media: float, desvio: float, valor: float, tipo: str = 'maior') -> float:
    """Calcula probabilidades usando a distribuição normal."""
    if tipo == 'maior': return 1 - stats.norm.cdf(valor, loc=media, scale=desvio)
    elif tipo == 'menor': return stats.norm.cdf(valor, loc=media, scale=desvio)
    elif tipo == 'entre': return stats.norm.cdf(valor[1], loc=media, scale=desvio) - stats.norm.cdf(valor[0], loc=media, scale=desvio)
    else: raise ValueError("Tipo deve ser 'maior', 'menor' ou 'entre'")

def calcular_dist_uniforme(a: float, b: float, valor: float, tipo: str = 'maior') -> float:
    """Calcula probabilidades usando a distribuição uniforme."""
    if tipo == 'maior':
        if valor > b: return 0
        elif valor < a: return 1
        else: return (b - valor) / (b - a)
    elif tipo == 'menor':
        if valor < a: return 0
        elif valor > b: return 1
        else: return (valor - a) / (b - a)
    elif tipo == 'entre':
        min_val, max_val = valor
        return calcular_dist_uniforme(a, b, max_val, 'menor') - calcular_dist_uniforme(a, b, min_val, 'menor')
    else: raise ValueError("Tipo deve ser 'maior', 'menor' ou 'entre'")

# --- FUNÇÕES DE PLOTAGEM (SEM ALTERAÇÕES) ---

def plotar_histograma(df: pd.DataFrame, coluna: str, titulo: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6)); sns.histplot(df[coluna], kde=True, bins=30)
    plt.axvline(df[coluna].mean(), color='red', linestyle='--', alpha=0.7, label=f'Média: {df[coluna].mean():.2f}')
    plt.axvline(df[coluna].median(), color='green', linestyle='--', alpha=0.7, label=f'Mediana: {df[coluna].median():.2f}')
    if titulo is None: titulo = f'Distribuição de {coluna}'
    titulo += f'\nMédia: {df[coluna].mean():.2f}, Mediana: {df[coluna].median():.2f}, Desvio: {df[coluna].std():.2f}'
    plt.title(titulo); plt.xlabel(coluna); plt.ylabel('Frequência'); plt.legend(); plt.tight_layout()

def plotar_boxplot(df: pd.DataFrame, coluna: str, titulo: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6)); sns.boxplot(x=df[coluna])
    if titulo is None: titulo = f'Boxplot de {coluna}'
    titulo += f'\nQ1: {df[coluna].quantile(0.25):.2f}, Mediana: {df[coluna].median():.2f}, Q3: {df[coluna].quantile(0.75):.2f}'
    plt.title(titulo); plt.xlabel(coluna); plt.tight_layout()

def plotar_regressao(df: pd.DataFrame, x: str, y: str, resultado_regressao: Dict[str, Any]) -> None:
    plt.figure(figsize=(10, 6)); sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    titulo = f'Regressão Linear: {x} vs {y}'
    titulo += f'\nEquação: {resultado_regressao["formula"]}'
    titulo += f'\nR²: {resultado_regressao["r_squared"]:.4f}, p-valor: {resultado_regressao["p_valor"]:.4e}'
    plt.title(titulo); plt.xlabel(x); plt.ylabel(y); plt.tight_layout()

def plotar_dist_binomial(n: int, p: float, titulo: Optional[str] = None) -> None:
    x = np.arange(0, n+1); pmf = stats.binom.pmf(x, n, p)
    plt.figure(figsize=(10, 6)); plt.bar(x, pmf, alpha=0.7); plt.plot(x, pmf, 'ro-', alpha=0.7)
    if titulo is None: titulo = f'Distribuição Binomial (n={n}, p={p:.2f})'
    plt.title(titulo); plt.xlabel('Número de Sucessos (k)'); plt.ylabel('Probabilidade P(X = k)')
    media = n * p; variancia = n * p * (1 - p)
    plt.figtext(0.15, 0.80, f'Média = n·p = {media:.2f}\nVariância = n·p·(1-p) = {variancia:.2f}', bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3); plt.tight_layout()

def plotar_dist_normal(media: float, desvio: float, coluna_dados: Optional[pd.Series] = None, titulo: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6)); x = np.linspace(media - 4*desvio, media + 4*desvio, 1000); pdf = stats.norm.pdf(x, loc=media, scale=desvio)
    plt.plot(x, pdf, 'r-', lw=2, label='Distribuição Normal Teórica')
    if coluna_dados is not None: sns.histplot(coluna_dados, kde=True, stat='density', alpha=0.5, label='Dados Reais')
    plt.axvline(media, color='blue', linestyle='--', alpha=0.7, label=f'Média: {media:.2f}')
    plt.axvline(media + desvio, color='green', linestyle='--', alpha=0.7, label=f'1 Desvio: {media+desvio:.2f}')
    plt.axvline(media - desvio, color='green', linestyle='--', alpha=0.7)
    if titulo is None: titulo = f'Distribuição Normal (μ={media:.2f}, σ={desvio:.2f})'
    plt.title(titulo); plt.xlabel('Valor'); plt.ylabel('Densidade de Probabilidade'); plt.legend()
    info_text = (f'Propriedades da Distribuição Normal:\n'f'• Média (μ) = {media:.2f}\n'f'• Desvio Padrão (σ) = {desvio:.2f}\n'f'• Probabilidade de estar entre μ±σ: 68.27%\n'f'• Probabilidade de estar entre μ±2σ: 95.45%\n'f'• Probabilidade de estar entre μ±3σ: 99.73%')
    plt.figtext(0.15, 0.80, info_text, bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3); plt.tight_layout()

def plotar_dist_uniforme(a: float, b: float, titulo: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6)); margem = (b - a) * 0.1; x = np.linspace(a - margem, b + margem, 1000); pdf = np.zeros_like(x)
    pdf[(x >= a) & (x <= b)] = 1.0 / (b - a); plt.plot(x, pdf, 'b-', lw=2); plt.fill_between(x, pdf, alpha=0.3, color='blue')
    plt.axvline(a, color='red', linestyle='--', alpha=0.7, label=f'Mínimo (a): {a:.2f}')
    plt.axvline(b, color='red', linestyle='--', alpha=0.7, label=f'Máximo (b): {b:.2f}')
    plt.axvline((a + b) / 2, color='green', linestyle='--', alpha=0.7, label=f'Média: {(a+b)/2:.2f}')
    if titulo is None: titulo = f'Distribuição Uniforme (a={a:.2f}, b={b:.2f})'
    plt.title(titulo); plt.xlabel('Valor'); plt.ylabel('Densidade de Probabilidade'); plt.legend()
    media = (a + b) / 2; variancia = ((b - a) ** 2) / 12; desvio = np.sqrt(variancia)
    info_text = (f'Propriedades da Distribuição Uniforme:\n'f'• Mínimo (a) = {a:.2f}\n'f'• Máximo (b) = {b:.2f}\n'f'• Média = (a+b)/2 = {media:.2f}\n'f'• Variância = (b-a)²/12 = {variancia:.2f}\n'f'• Desvio Padrão = {desvio:.2f}\n'f'• Probabilidade para qualquer intervalo: comprimento/amplitude')
    plt.figtext(0.15, 0.80, info_text, bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3); plt.tight_layout()


# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO DA ANÁLISE ---

def executar_analise_completa():
    """Função principal que executa toda a análise estatística."""
    print("=== ANÁLISE ESTATÍSTICA DE DADOS DE VENDAS ===\n")
    
    # 1. Configuração e Carregamento
    configurar_visualizacao()
    nome_arquivo_excel = 'trabalho_estatistica.xlsx'
    df = carregar_dados(nome_arquivo_excel)
    df_limpo = preparar_dados(df)
    
    # 2. Estatísticas Descritivas
    # Objetivo: Resumir as principais características dos dados.
    print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    
    # Análise do valor total da venda
    # Média: Valor médio de cada venda.
    # Mediana: Valor central que divide as vendas em duas metades (50% abaixo, 50% acima).
    # Desvio Padrão: Mede o quão dispersos os valores das vendas estão em relação à média.
    # Quartis (Q1, Q3): Dividem os dados em quatro partes. Mostram a faixa de preço dos 50% centrais das vendas.
    print("\n[1] Análise do Total da Venda:")
    stats_total_venda = calcular_estatisticas_descritivas(df_limpo, 'total_venda')
    for stat, valor in stats_total_venda.items():
        print(f"  • {stat.replace('_', ' ').title()}: {valor:.2f}")
    
    # Análise do valor unitário
    print("\n[2] Análise do Valor Unitário:")
    stats_valor_unitario = calcular_estatisticas_descritivas(df_limpo, 'valor_unitario')
    for stat, valor in stats_valor_unitario.items():
        print(f"  • {stat.replace('_', ' ').title()}: {valor:.2f}")
    
    # Análise da quantidade
    print("\n[3] Análise da Quantidade:")
    stats_quantidade = calcular_estatisticas_descritivas(df_limpo, 'quantity')
    for stat, valor in stats_quantidade.items():
        print(f"  • {stat.replace('_', ' ').title()}: {valor:.2f}")
    
    # Moda do produto mais vendido
    # Moda: O valor que aparece com mais frequência. Aqui, identifica o ID do produto mais vendido.
    print("\n[4] Produto(s) mais vendido(s) (Moda):")
    moda_produto = df_limpo['product_id'].mode()
    if not moda_produto.empty:
        moda_list = moda_produto.tolist()
        print(f"  • Produto(s): {', '.join(map(str, moda_list))}")
        freq_moda = df_limpo['product_id'].value_counts().iloc[0]
        pct_moda = (freq_moda / len(df_limpo)) * 100
        print(f"  • Frequência: {freq_moda} ocorrências ({pct_moda:.2f}% do total)")
    else:
        print("  • Não foi possível determinar a moda.")
    
    # 3. Análise de Normalidade
    # Objetivo: Verificar se os dados seguem uma distribuição Normal (curva de sino).
    # Teste de Shapiro-Wilk: Um dos testes mais comuns para normalidade.
    # P-valor > 0.05: Indica que os dados provavelmente seguem uma distribuição normal.
    # P-valor < 0.05: Indica que os dados provavelmente NÃO seguem uma distribuição normal.
    print("\n=== ANÁLISE DE NORMALIDADE ===")
    
    print("\n[1] Teste para Total da Venda:")
    stat, p_valor, eh_normal = testar_normalidade(df_limpo, 'total_venda')
    print(f"  • Estatística do teste Shapiro-Wilk: {stat:.4f}")
    print(f"  • P-valor: {p_valor:.4e}")
    print(f"  • Conclusão: Os dados {'seguem' if eh_normal else 'NÃO seguem'} uma distribuição normal.")
    
    print("\n[2] Teste para Valor Unitário:")
    stat, p_valor, eh_normal = testar_normalidade(df_limpo, 'valor_unitario')
    print(f"  • Estatística do teste Shapiro-Wilk: {stat:.4f}")
    print(f"  • P-valor: {p_valor:.4e}")
    print(f"  • Conclusão: Os dados {'seguem' if eh_normal else 'NÃO seguem'} uma distribuição normal.")
    
    # 4. Análise de Correlação e Regressão
    # Objetivo: Entender a relação entre duas variáveis numéricas.
    print("\n=== CORRELAÇÃO E REGRESSÃO ===")
    
    # Correlação entre quantidade e total_venda
    # Coeficiente de Pearson (r): Mede a força e a direção da relação linear (de -1 a 1).
    # R²: Indica a porcentagem da variação de uma variável que é explicada pela outra.
    print("\n[1] Correlação entre Quantidade e Total da Venda:")
    corr_result = calcular_correlacao(df_limpo, 'quantity', 'total_venda')
    print(f"  • Coeficiente de Pearson: {corr_result['coeficiente']:.4f}")
    print(f"  • Interpretação: {corr_result['interpretacao']}")
    print(f"  • R² (Coeficiente de determinação): {corr_result['r_squared']:.4f}")
    print(f"  • P-valor: {corr_result['p_valor']:.4e}")
    print(f"  • Correlação {'é' if corr_result['significativa'] else 'NÃO é'} estatisticamente significativa.")
    
    # Regressão linear (Estatística Preditiva)
    # Objetivo: Criar um modelo (equação) para prever o valor de uma variável com base em outra.
    # Equação: Mostra como calcular o 'total_venda' a partir da 'quantity'.
    print("\n[2] Regressão Linear (Quantidade → Total da Venda):")
    reg_result = realizar_regressao(df_limpo, 'quantity', 'total_venda')
    print(f"  • Equação: {reg_result['formula']}")
    print(f"  • R²: {reg_result['r_squared']:.4f}")
    print(f"  • R² Ajustado: {reg_result['r_squared_adj']:.4f}")
    print(f"  • P-valor: {reg_result['p_valor']:.4e}")
    print(f"  • Modelo {'é' if reg_result['significativo'] else 'NÃO é'} estatisticamente significativo.")
    
    # AJUSTE ADICIONADO: Imprime a tabela de sumário completa da regressão.
    # Esta tabela mostra detalhes cruciais: coeficientes, erros, e intervalos de confiança,
    # permitindo uma análise muito mais profunda do modelo preditivo.
    print("\n  • Sumário Completo do Modelo de Regressão:")
    print(reg_result['modelo'].summary())
    
    # 5. Testes de Hipóteses
    # Objetivo: Usar dados de uma amostra para fazer inferências sobre uma população.
    # Teste t: Compara as médias de dois grupos para ver se a diferença entre elas é estatisticamente significativa.
    print("\n=== TESTES DE HIPÓTESES ===")
    
    print("\n[1] Teste t: Comparação do valor unitário de vendas altas vs. baixas")
    mediana_total = df_limpo['total_venda'].median()
    grupo_vendas_altas = df_limpo[df_limpo['total_venda'] > mediana_total]['valor_unitario']
    grupo_vendas_baixas = df_limpo[df_limpo['total_venda'] <= mediana_total]['valor_unitario']
    
    resultado_teste_t = realizar_teste_t(df_limpo, grupo_vendas_altas, grupo_vendas_baixas, 'valor_unitario')
    
    # Hipótese Nula (H0): A suposição inicial de que não há efeito ou diferença (ex: as médias são iguais).
    # Hipótese Alternativa (H1): A suposição que queremos provar (ex: as médias são diferentes).
    # P-valor < 0.05: Rejeitamos H0, a diferença é significativa.
    print(f"  • Hipótese Nula (H0): Não há diferença no valor unitário médio entre vendas altas e baixas.")
    print(f"  • Hipótese Alternativa (H1): Há diferença significativa no valor unitário médio.")
    print(f"  • Estatística t: {resultado_teste_t['estatistica_t']:.4f}")
    print(f"  • P-valor: {resultado_teste_t['p_valor']:.4e}")
    print(f"  • Média do valor unitário em vendas altas: {resultado_teste_t['media_grupo1']:.2f}")
    print(f"  • Média do valor unitário em vendas baixas: {resultado_teste_t['media_grupo2']:.2f}")
    print(f"  • Diferença: {resultado_teste_t['diferenca_medias']:.2f}")
    print(f"  • Conclusão: {'Rejeitamos' if resultado_teste_t['diferenca_significativa'] else 'Não rejeitamos'} a hipótese nula.")
    
    # 6. Análise de Probabilidade
    # Objetivo: Calcular a chance de certos eventos ocorrerem com base nos dados existentes.
    print("\n=== ANÁLISE DE PROBABILIDADE ===")
    
    print("\n[1] Probabilidades Empíricas (Total da Venda):")
    prob_result = calcular_probabilidades(df_limpo, 'total_venda')
    print(f"  • Probabilidade de uma venda ser maior que a média: {prob_result['prob_maior_media']:.2%}")
    print(f"  • Probabilidade de uma venda ser maior que a mediana: {prob_result['prob_maior_mediana']:.2%}")
    print(f"  • Probabilidade de uma venda estar entre Q1 e Q3: {prob_result['prob_entre_q1_q3']:.2%}")
    
    # 7. Distribuições Estatísticas Teóricas
    # Objetivo: Usar modelos teóricos para calcular probabilidades.
    print("\n=== DISTRIBUIÇÕES ESTATÍSTICAS ===")
    
    # A) Distribuição Binomial
    # Usada quando temos um número fixo de tentativas (n) com apenas dois resultados possíveis (sucesso/fracasso).
    # Cenário: Qual a chance de, em 5 vendas, exatamente 3 serem acima da média?
    print("\n[1] Distribuição Binomial:")
    p_sucesso = prob_result['prob_maior_media']
    n_tentativas = 5
    k_sucessos = 3
    prob_binomial = calcular_dist_binomial(p_sucesso, n_tentativas, k_sucessos)
    print(f"  • Cenário: Probabilidade de obter exatamente {k_sucessos} vendas acima da média em {n_tentativas} tentativas")
    print(f"  • Probabilidade de sucesso (venda > média): {p_sucesso:.2%}")
    print(f"  • Resultado P(X = {k_sucessos}): {prob_binomial:.2%}")
    
    # B) Distribuição Normal
    # Usa a 'curva de sino' para calcular probabilidades para variáveis contínuas,
    # assumindo que os dados se comportam de acordo com a média (μ) e desvio padrão (σ).
    print("\n[2] Distribuição Normal (aplicada ao Total da Venda):")
    media = stats_total_venda['media']
    desvio = stats_total_venda['desvio_padrao']
    print(f"  • Usando Média (μ) = {media:.2f} e Desvio Padrão (σ) = {desvio:.2f} dos dados.")
    valor_corte1 = media + desvio
    prob_normal1 = calcular_dist_normal(media, desvio, valor_corte1, 'maior')
    print(f"  • Exemplo: Probabilidade de uma venda ser maior que a média + 1 desvio padrão ({valor_corte1:.2f})")
    print(f"  • Resultado P(X > {valor_corte1:.2f}) = {prob_normal1:.2%}")
    
    # C) Distribuição Uniforme
    # Usada quando todos os resultados em um intervalo [a, b] são igualmente prováveis.
    # Cenário: Se qualquer valor de venda entre o mínimo e o máximo fosse igualmente possível.
    print("\n[3] Distribuição Uniforme (aplicada à faixa de preços):")
    min_preco = stats_total_venda['minimo']
    max_preco = stats_total_venda['maximo']
    valor_medio = (min_preco + max_preco) / 2
    prob_unif1 = calcular_dist_uniforme(min_preco, max_preco, valor_medio, 'maior')
    print(f"  • Cenário: Preços uniformemente distribuídos entre {min_preco:.2f} e {max_preco:.2f}")
    print(f"  • Exemplo: Probabilidade de uma venda ser maior que o valor médio da faixa ({valor_medio:.2f})")
    print(f"  • Resultado P(X > {valor_medio:.2f}) = {prob_unif1:.2%}")
    
    # 8. Visualizações
    print("\n=== GERANDO VISUALIZAÇÕES ===")
    print("Gerando histogramas, boxplots, gráficos de distribuição e regressão...")
    
    plotar_histograma(df_limpo, 'total_venda', 'Distribuição do Valor Total da Venda')
    plotar_boxplot(df_limpo, 'total_venda', 'Boxplot do Valor Total da Venda')
    plotar_regressao(df_limpo, 'quantity', 'total_venda', reg_result)
    plotar_dist_binomial(n_tentativas, p_sucesso, f'Distribuição Binomial (n={n_tentativas}, p={p_sucesso:.2f})')
    plotar_dist_normal(media, desvio, df_limpo['total_venda'], 'Distribuição Normal vs. Dados Reais (Total da Venda)')
    plotar_dist_uniforme(min_preco, max_preco, f'Distribuição Uniforme da Faixa de Preços')
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    print("Todos os gráficos foram gerados. Utilize plt.show() para visualizá-los.")
    plt.show()

# Se este arquivo for executado diretamente, executa a análise completa
if __name__ == "__main__":
    executar_analise_completa()