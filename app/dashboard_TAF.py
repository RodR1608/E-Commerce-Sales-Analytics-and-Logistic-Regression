
from pathlib import Path
import html

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# DASHBOARD TAF V2.3
# E-COMMERCE SALES ANALYTICS
# BUSINESS DATA & AI STRATEGY
# ============================================================

st.set_page_config(
    page_title="TAF | E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================

BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "processed"
TABLAS = BASE / "outputs" / "tablas"


# ============================================================
# 2. PALETA CORPORATIVA
# ============================================================

BG = "#041333"
SIDEBAR = "#06183B"
CARD = "#091F47"
CARD_ALT = "#0D2854"
BORDER = "#17345E"

CYAN = "#00D4E8"
TURQUOISE = "#12B8CE"
BLUE = "#2388E8"
GREEN = "#27D6B3"

WHITE = "#F5FAFF"
MUTED = "#9BB2CE"

ORANGE = "#F6A85F"
RED = "#FF7185"

PALETTE = [
    CYAN,
    BLUE,
    GREEN,
    TURQUOISE,
    ORANGE,
    "#8D9BFF",
    RED
]

STATUS_COLORS = {
    "Completed": CYAN,
    "Cancelled": RED
}


# ============================================================
# 3. CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BG};
        color: {WHITE};
    }}

    header[data-testid="stHeader"] {{
        background-color: {BG};
    }}

    section[data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] * {{
        color: {WHITE};
    }}

    h1, h2, h3, h4 {{
        color: {WHITE};
    }}

    .block-container {{
        padding-top: 1.4rem;
        max-width: 1600px;
    }}

    .hero {{
        background: linear-gradient(
            110deg,
            {CARD},
            {SIDEBAR}
        );
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 23px 26px;
        margin-bottom: 17px;
    }}

    .hero-title {{
        color: {WHITE};
        font-size: 29px;
        font-weight: 750;
    }}

    .hero-sub {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 6px;
    }}

    .filter-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 4px;
        margin-bottom: 6px;
    }}

    .filter-icon {{
        color: {CYAN};
        font-size: 23px;
    }}

    .filter-title {{
        color: {WHITE};
        font-size: 17px;
        font-weight: 700;
    }}

    .filter-sub {{
        color: {MUTED};
        font-size: 11px;
        letter-spacing: 1px;
    }}

    .filter-footer {{
        color: {MUTED};
        font-size: 12px;
        padding-top: 5px;
    }}

    .filter-count {{
        display: inline-block;
        color: {CYAN};
        background: {CARD_ALT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        font-weight: 650;
    }}

    .kpi {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-top: 3px solid {CYAN};
        border-radius: 12px;
        padding: 17px;
        min-height: 118px;
        margin-bottom: 12px;
    }}

    .kpi-label {{
        color: {MUTED};
        font-size: 12px;
        font-weight: 650;
        text-transform: uppercase;
    }}

    .kpi-value {{
        color: {WHITE};
        font-size: 27px;
        font-weight: 750;
        margin-top: 8px;
    }}

    .kpi-note {{
        color: {MUTED};
        font-size: 11px;
        margin-top: 6px;
    }}

    .insight {{
        background: {CARD};
        border-left: 3px solid {CYAN};
        border-radius: 9px;
        padding: 13px 16px;
        color: {WHITE};
        margin: 15px 0;
        font-size: 13px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {BORDER};
    }}

    div[data-testid="stButton"] button {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        background-color: {CARD_ALT};
        color: {WHITE};
        font-weight: 600;
    }}

    div[data-testid="stButton"] button:hover {{
        border-color: {CYAN};
        color: {CYAN};
    }}

    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: {CYAN};
        color: {BG};
        border-color: {CYAN};
    }}

    div[data-testid="stSelectbox"] > div > div {{
        background-color: {CARD_ALT};
        border-color: {BORDER};
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    hr {{
        border-color: {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. CARGA DE ARCHIVOS
# ============================================================

@st.cache_data
def leer_csv(ruta):

    if ruta.exists():
        return pd.read_csv(
            ruta,
            low_memory=False
        )

    return None


df = leer_csv(
    DATA / "dataset_analitico_validado.csv"
)

if df is None:

    st.error(
        "No se encontró el archivo "
        "data/processed/dataset_analitico_validado.csv"
    )

    st.stop()


metricas = leer_csv(
    TABLAS / "41_metricas_modelo.csv"
)

confusion = leer_csv(
    TABLAS / "42_matriz_confusion.csv"
)

coeficientes = leer_csv(
    TABLAS / "43_coeficientes_modelo.csv"
)

predicciones = leer_csv(
    TABLAS / "44_predicciones_test.csv"
)

comparacion = leer_csv(
    TABLAS / "46_comparacion_modelos_resumen.csv"
)

umbrales = leer_csv(
    TABLAS / "47_evaluacion_umbrales.csv"
)

umbral_seleccionado = leer_csv(
    TABLAS / "48_umbral_seleccionado.csv"
)


# ============================================================
# 5. VALIDACIÓN Y PREPARACIÓN DEL DATASET
# ============================================================

obligatorias = [
    "order_id",
    "order_date",
    "target"
]

faltantes = [
    columna
    for columna in obligatorias
    if columna not in df.columns
]

if faltantes:

    st.error(
        f"Columnas faltantes: {faltantes}"
    )

    st.stop()


df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

df = df.dropna(
    subset=["order_date", "target"]
).copy()

df = df[
    df["target"].isin([0, 1])
].copy()

if df.empty:

    st.error(
        "No existen órdenes válidas."
    )

    st.stop()


df["target"] = df["target"].astype(int)

df["anio"] = df["order_date"].dt.year

df["mes"] = (
    df["order_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

df["mes_num"] = df["order_date"].dt.month

df["dia_semana"] = (
    df["order_date"].dt.dayofweek
)


columnas_numericas = [
    "customer_age",
    "customer_order_count",
    "customer_recency",
    "order_quantity",
    "order_gross_sales",
    "shipping_cost_ratio"
]

for columna in columnas_numericas:

    if columna in df.columns:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )


# ============================================================
# 6. FUNCIONES VISUALES
# ============================================================

def tema(fig, titulo="", altura=370):

    fig.update_layout(
        title=dict(
            text=titulo,
            font=dict(
                size=16,
                color=WHITE
            )
        ),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=WHITE),
        colorway=PALETTE,
        height=altura,
        margin=dict(
            l=35,
            r=25,
            t=65,
            b=40
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED)
        ),
        hoverlabel=dict(
            bgcolor=CARD_ALT,
            font_color=WHITE
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED)
    )

    fig.update_yaxes(
        gridcolor=BORDER,
        zeroline=False,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED)
    )

    return fig


def mostrar(fig):

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def titulo_pagina(titulo, subtitulo):

    contenido = (
        '<div class="hero">'
        f'<div class="hero-title">{html.escape(str(titulo))}</div>'
        f'<div class="hero-sub">{html.escape(str(subtitulo))}</div>'
        '</div>'
    )

    st.markdown(
        contenido,
        unsafe_allow_html=True
    )


def tarjeta(nombre, valor, nota=""):

    # ====================================================
    # CORRECCIÓN V2.3
    # HTML en una sola cadena sin saltos ni indentaciones.
    # Evita que Streamlit muestre etiquetas </div>.
    # ====================================================

    contenido = (
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(str(nombre))}</div>'
        f'<div class="kpi-value">{html.escape(str(valor))}</div>'
        f'<div class="kpi-note">{html.escape(str(nota))}</div>'
        '</div>'
    )

    st.markdown(
        contenido,
        unsafe_allow_html=True
    )


def aviso(mensaje):

    contenido = (
        '<div class="insight">'
        f'{html.escape(str(mensaje))}'
        '</div>'
    )

    st.markdown(
        contenido,
        unsafe_allow_html=True
    )


# ============================================================
# 7. FUNCIONES ANALÍTICAS
# ============================================================

def tasa_cancelacion(datos):

    if datos.empty:
        return 0.0

    return float(
        datos["target"].mean() * 100
    )


def resumen_grupo(datos, columna):

    resultado = (
        datos.groupby(
            columna,
            observed=True
        )
        .agg(
            ordenes=("target", "size"),
            canceladas=("target", "sum")
        )
        .reset_index()
    )

    if resultado.empty:

        resultado["tasa"] = pd.Series(
            dtype=float
        )

        return resultado

    resultado["tasa"] = (
        resultado["canceladas"]
        / resultado["ordenes"]
        * 100
    )

    return resultado


def grafico_tasa(
    datos,
    columna,
    titulo,
    ordenar=False
):

    if columna not in datos.columns:

        st.info(
            f"No disponible: {columna}"
        )

        return

    agrupado = resumen_grupo(
        datos.dropna(subset=[columna]),
        columna
    )

    if agrupado.empty:

        st.info(
            "Sin datos para este gráfico."
        )

        return

    # ====================================================
    # CORRECCIÓN V2.3
    # Convierte pandas.Interval en texto antes de Plotly.
    # No modifica los cálculos estadísticos.
    # ====================================================

    agrupado[columna] = (
        agrupado[columna].astype(str)
    )

    if ordenar:

        agrupado = agrupado.sort_values(
            "tasa",
            ascending=False
        )

    categorias = agrupado[columna].tolist()

    fig = px.bar(
        agrupado,
        x=columna,
        y="tasa",
        color_discrete_sequence=[CYAN],
        category_orders={
            columna: categorias
        },
        hover_data=[
            "ordenes",
            "canceladas"
        ],
        labels={
            columna: "",
            "tasa": "Cancelación (%)"
        }
    )

    fig.update_traces(
        marker_line_width=0
    )

    mostrar(
        tema(
            fig,
            titulo
        )
    )


def grafico_box(datos, columna, titulo):

    if columna not in datos.columns:
        return

    visual = datos[
        ["target", columna]
    ].dropna().copy()

    if visual.empty:
        return

    visual["Estado"] = (
        visual["target"].map({
            0: "Completed",
            1: "Cancelled"
        })
    )

    fig = px.box(
        visual,
        x="Estado",
        y=columna,
        color="Estado",
        color_discrete_map=STATUS_COLORS,
        points=False
    )

    mostrar(
        tema(
            fig,
            titulo
        )
    )


def obtener_metrica(nombre):

    if metricas is None or metricas.empty:
        return None

    # Formato vertical
    if {
        "metrica",
        "valor"
    }.issubset(metricas.columns):

        registros = metricas.loc[
            metricas["metrica"]
            .astype(str)
            .str.lower()
            == nombre.lower()
        ]

        if not registros.empty:

            return float(
                registros.iloc[0]["valor"]
            )

    # Formato horizontal
    if nombre in metricas.columns:

        return float(
            metricas.iloc[0][nombre]
        )

    return None


def formato_metrica(nombre, porcentaje=False):

    valor = obtener_metrica(nombre)

    if valor is None:
        return "N/D"

    if porcentaje:
        return f"{valor:.2%}"

    return f"{valor:.4f}"


# ============================================================
# 8. NAVEGACIÓN LATERAL
# ============================================================

st.sidebar.markdown(
    "## ◈ E-COMMERCE"
)

st.sidebar.caption(
    "BUSINESS DATA & AI STRATEGY"
)

st.sidebar.divider()

pagina = st.sidebar.radio(
    "MÓDULOS",
    [
        "01 · Resumen ejecutivo",
        "02 · Análisis temporal",
        "03 · Clientes y segmentos",
        "04 · Órdenes y operaciones",
        "05 · Evaluación predictiva",
        "06 · Explorador de riesgo"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.caption(
    "TAF | Metodología CRISP-DM"
)

st.sidebar.caption(
    "Dashboard de análisis de cancelaciones"
)


# ============================================================
# 9. VALORES DISPONIBLES PARA FILTROS
# ============================================================

ANIOS = sorted(
    int(anio)
    for anio in df["anio"].dropna().unique()
)

CANALES = (
    sorted(
        df["sales_channel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "sales_channel" in df.columns
    else []
)

SEGMENTOS = (
    sorted(
        df["customer_segment"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "customer_segment" in df.columns
    else []
)


# ============================================================
# 10. ESTADO DE FILTROS
# ============================================================

if "filtro_anios" not in st.session_state:

    st.session_state.filtro_anios = ANIOS.copy()

if "filtro_canal" not in st.session_state:

    st.session_state.filtro_canal = "Todos"

if "filtro_segmento" not in st.session_state:

    st.session_state.filtro_segmento = "Todos"


def alternar_anio(anio):

    actuales = (
        st.session_state.filtro_anios.copy()
    )

    if anio in actuales:

        actuales.remove(anio)

    else:

        actuales.append(anio)

    st.session_state.filtro_anios = (
        sorted(actuales)
    )


def restablecer_filtros():

    st.session_state.filtro_anios = (
        ANIOS.copy()
    )

    st.session_state.filtro_canal = "Todos"

    st.session_state.filtro_segmento = "Todos"


# ============================================================
# 11. TÍTULO DE CADA VISTA
# ============================================================

TITULOS = {
    "01": (
        "Resumen ejecutivo",
        "Indicadores generales de órdenes y cancelaciones"
    ),
    "02": (
        "Análisis temporal",
        "Evolución histórica y patrones temporales"
    ),
    "03": (
        "Clientes y segmentos",
        "Comportamiento de clientes y antecedentes de compra"
    ),
    "04": (
        "Órdenes y operaciones",
        "Cantidad de productos, monto bruto y costos de envío"
    ),
    "05": (
        "Evaluación predictiva",
        "Resultados de regresión logística y validación"
    ),
    "06": (
        "Explorador de riesgo",
        "Probabilidades estimadas sobre el conjunto Test"
    )
}

codigo_pagina = pagina[:2]

titulo_pagina(
    *TITULOS[codigo_pagina]
)


# ============================================================
# 12. PANEL EJECUTIVO DE FILTROS
# ============================================================

with st.container(border=True):

    st.markdown(
        (
            '<div class="filter-header">'
            '<div class="filter-icon">☷</div>'
            '<div>'
            '<div class="filter-title">Panel de control</div>'
            '<div class="filter-sub">FILTROS DE ANÁLISIS</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True
    )

    st.caption(
        "Periodo de análisis"
    )

    columnas_anios = st.columns(
        max(1, len(ANIOS))
    )

    for columna_visual, anio in zip(
        columnas_anios,
        ANIOS
    ):

        seleccionado = (
            anio in st.session_state.filtro_anios
        )

        with columna_visual:

            st.button(
                str(anio),
                key=f"boton_anio_{anio}",
                type=(
                    "primary"
                    if seleccionado
                    else "secondary"
                ),
                use_container_width=True,
                on_click=alternar_anio,
                args=(anio,)
            )

    st.write("")

    c1, c2, c3 = st.columns(
        [2.2, 2.2, 1.2]
    )

    with c1:

        st.selectbox(
            "Canal de venta",
            options=["Todos"] + CANALES,
            key="filtro_canal"
        )

    with c2:

        st.selectbox(
            "Segmento de cliente",
            options=["Todos"] + SEGMENTOS,
            key="filtro_segmento"
        )

    with c3:

        st.write("")

        st.button(
            "↻ Restablecer",
            on_click=restablecer_filtros,
            use_container_width=True
        )


# ============================================================
# 13. APLICACIÓN DE LOS FILTROS
# ============================================================

filtrado = df[
    df["anio"].isin(
        st.session_state.filtro_anios
    )
].copy()


if (
    "sales_channel" in filtrado.columns
    and st.session_state.filtro_canal != "Todos"
):

    filtrado = filtrado[
        filtrado["sales_channel"]
        == st.session_state.filtro_canal
    ]


if (
    "customer_segment" in filtrado.columns
    and st.session_state.filtro_segmento != "Todos"
):

    filtrado = filtrado[
        filtrado["customer_segment"]
        == st.session_state.filtro_segmento
    ]


total = len(filtrado)


st.markdown(
    (
        '<div class="filter-footer">'
        '<span class="filter-count">'
        f'{total:,} órdenes seleccionadas'
        '</span>'
        '&nbsp;&nbsp;'
        'Los filtros se aplican a los análisis descriptivos.'
        '</div>'
    ),
    unsafe_allow_html=True
)

st.write("")


if filtrado.empty:

    st.warning(
        "No existen registros con los filtros "
        "seleccionados."
    )

    st.stop()


# ============================================================
# 14. INDICADORES GENERALES
# ============================================================

canceladas = int(
    filtrado["target"].sum()
)

completadas = total - canceladas

tasa = tasa_cancelacion(filtrado)


# ============================================================
# VISTA 01 - RESUMEN EJECUTIVO
# ============================================================

if codigo_pagina == "01":

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        tarjeta(
            "Órdenes analizadas",
            f"{total:,}",
            "Completed y Cancelled"
        )

    with c2:

        tarjeta(
            "Órdenes completadas",
            f"{completadas:,}",
            "Órdenes finalizadas"
        )

    with c3:

        tarjeta(
            "Órdenes canceladas",
            f"{canceladas:,}",
            "Órdenes con target = 1"
        )

    with c4:

        tarjeta(
            "Tasa de cancelación",
            f"{tasa:.2f}%",
            "Canceladas / órdenes analizadas"
        )

    st.write("")

    evolucion = resumen_grupo(
        filtrado,
        "mes"
    ).sort_values("mes")

    fig = px.line(
        evolucion,
        x="mes",
        y="tasa",
        markers=True,
        color_discrete_sequence=[CYAN],
        labels={
            "mes": "Mes",
            "tasa": "Tasa (%)"
        }
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=6)
    )

    mostrar(
        tema(
            fig,
            "Evolución mensual de cancelaciones"
        )
    )

    izquierda, derecha = st.columns(2)

    with izquierda:

        fig = go.Figure(
            go.Pie(
                labels=[
                    "Completed",
                    "Cancelled"
                ],
                values=[
                    completadas,
                    canceladas
                ],
                hole=0.68,
                marker=dict(
                    colors=[
                        CYAN,
                        RED
                    ]
                ),
                textinfo="percent+label"
            )
        )

        mostrar(
            tema(
                fig,
                "Distribución de órdenes"
            )
        )

    with derecha:

        if "sales_channel" in filtrado.columns:

            grafico_tasa(
                filtrado,
                "sales_channel",
                "Tasa de cancelación por canal",
                ordenar=True
            )

    if "order_gross_sales" in filtrado.columns:

        monto_bruto = (
            filtrado["order_gross_sales"].sum()
        )

        aviso(
            f"Monto bruto registrado: {monto_bruto:,.2f}. "
            "Incluye órdenes completadas y canceladas. "
            "No representa ingresos efectivamente realizados."
        )


# ============================================================
# VISTA 02 - ANÁLISIS TEMPORAL
# ============================================================

elif codigo_pagina == "02":

    anuales = resumen_grupo(
        filtrado,
        "anio"
    ).sort_values("anio")

    izquierda, derecha = st.columns(2)

    with izquierda:

        fig = px.bar(
            anuales,
            x="anio",
            y="ordenes",
            color_discrete_sequence=[BLUE]
        )

        mostrar(
            tema(
                fig,
                "Volumen anual de órdenes"
            )
        )

    with derecha:

        fig = px.line(
            anuales,
            x="anio",
            y="tasa",
            markers=True,
            color_discrete_sequence=[GREEN]
        )

        mostrar(
            tema(
                fig,
                "Tasa anual de cancelación"
            )
        )

    mensuales = resumen_grupo(
        filtrado,
        "mes"
    ).sort_values("mes")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=mensuales["mes"],
            y=mensuales["ordenes"],
            marker_color=BLUE,
            name="Órdenes"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mensuales["mes"],
            y=mensuales["canceladas"],
            mode="lines+markers",
            line=dict(
                color=CYAN,
                width=3
            ),
            name="Canceladas"
        )
    )

    mostrar(
        tema(
            fig,
            "Evolución mensual de órdenes y cancelaciones"
        )
    )

    matriz_temporal = (
        filtrado
        .groupby(["anio", "mes_num"])["target"]
        .mean()
        .mul(100)
        .unstack()
        .reindex(columns=range(1, 13))
    )

    fig = go.Figure(
        go.Heatmap(
            z=matriz_temporal.to_numpy(),
            x=[
                "Ene", "Feb", "Mar", "Abr",
                "May", "Jun", "Jul", "Ago",
                "Sep", "Oct", "Nov", "Dic"
            ],
            y=matriz_temporal.index.astype(str),
            colorscale=[
                [0, BG],
                [0.5, BLUE],
                [1, CYAN]
            ],
            hovertemplate=(
                "Año: %{y}<br>"
                "Mes: %{x}<br>"
                "Cancelación: %{z:.2f}%"
                "<extra></extra>"
            )
        )
    )

    mostrar(
        tema(
            fig,
            "Mapa de calor de cancelación por mes"
        )
    )

    dias = filtrado.copy()

    mapa_dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo"
    }

    dias["nombre_dia"] = (
        dias["dia_semana"].map(mapa_dias)
    )

    dias["nombre_dia"] = pd.Categorical(
        dias["nombre_dia"],
        categories=list(mapa_dias.values()),
        ordered=True
    )

    grafico_tasa(
        dias,
        "nombre_dia",
        "Cancelación por día de la semana"
    )

    aviso(
        "Los patrones temporales son descriptivos. "
        "No constituyen evidencia estadística de estacionalidad."
    )


# ============================================================
# VISTA 03 - CLIENTES Y SEGMENTOS
# ============================================================

elif codigo_pagina == "03":

    if "customer_segment" in filtrado.columns:

        grafico_tasa(
            filtrado,
            "customer_segment",
            "Cancelación por segmento de cliente",
            ordenar=True
        )

    if "customer_age" in filtrado.columns:

        edad = filtrado.copy()

        edad["grupo_edad"] = pd.cut(
            edad["customer_age"],
            bins=[
                -np.inf,
                25,
                35,
                45,
                55,
                65,
                np.inf
            ],
            labels=[
                "Hasta 25",
                "26–35",
                "36–45",
                "46–55",
                "56–65",
                "Más de 65"
            ]
        )

        grafico_tasa(
            edad,
            "grupo_edad",
            "Cancelación por grupo de edad"
        )

    if "customer_order_count" in filtrado.columns:

        clientes = filtrado.copy()

        clientes["rango_frecuencia"] = pd.cut(
            clientes["customer_order_count"],
            bins=[
                -np.inf,
                0,
                1,
                3,
                5,
                10,
                np.inf
            ],
            labels=[
                "Sin anteriores",
                "1 anterior",
                "2–3 anteriores",
                "4–5 anteriores",
                "6–10 anteriores",
                "Más de 10"
            ]
        )

        grafico_tasa(
            clientes,
            "rango_frecuencia",
            "Cancelación según órdenes anteriores"
        )

    izquierda, derecha = st.columns(2)

    with izquierda:

        if "customer_recency" in filtrado.columns:

            grafico_box(
                filtrado,
                "customer_recency",
                "Recencia histórica por estado"
            )

    with derecha:

        if "customer_order_count" in filtrado.columns:

            clientes = filtrado.copy()

            clientes["tipo_cliente"] = np.where(
                clientes["customer_order_count"] == 0,
                "Sin compras anteriores",
                "Con compras anteriores"
            )

            grafico_tasa(
                clientes,
                "tipo_cliente",
                "Cancelación según historial de compras"
            )

    aviso(
        "La frecuencia y recencia se calcularon "
        "a partir de órdenes anteriores a la fecha "
        "de cada transacción. Los resultados muestran "
        "asociaciones, no relaciones causales."
    )


# ============================================================
# VISTA 04 - ÓRDENES Y OPERACIONES
# ============================================================

elif codigo_pagina == "04":

    variables = [
        (
            "order_quantity",
            "Cantidad de productos"
        ),
        (
            "order_gross_sales",
            "Monto bruto"
        ),
        (
            "shipping_cost_ratio",
            "Costo relativo de envío"
        )
    ]

    disponibles = [
        (columna, etiqueta)
        for columna, etiqueta in variables
        if columna in filtrado.columns
    ]

    if disponibles:

        columnas = st.columns(
            len(disponibles)
        )

        for columna_visual, (columna, etiqueta) in zip(
            columnas,
            disponibles
        ):

            valor = (
                filtrado[columna].median()
            )

            with columna_visual:

                tarjeta(
                    etiqueta,
                    (
                        f"{valor:,.2f}"
                        if pd.notna(valor)
                        else "N/D"
                    ),
                    "Mediana de las órdenes filtradas"
                )

    st.write("")

    if "order_quantity" in filtrado.columns:

        cantidad = filtrado.copy()

        cantidad["rango_cantidad"] = pd.cut(
            cantidad["order_quantity"],
            bins=[
                -np.inf,
                1,
                2,
                3,
                5,
                10,
                np.inf
            ],
            labels=[
                "Hasta 1",
                "2",
                "3",
                "4–5",
                "6–10",
                "Más de 10"
            ]
        )

        grafico_tasa(
            cantidad,
            "rango_cantidad",
            "Cancelación por cantidad de productos"
        )

    if "order_gross_sales" in filtrado.columns:

        ventas = filtrado.dropna(
            subset=["order_gross_sales"]
        ).copy()

        if (
            ventas["order_gross_sales"].nunique()
            > 1
        ):

            ventas["rango_valor"] = pd.qcut(
                ventas["order_gross_sales"],
                q=5,
                duplicates="drop"
            )

            grafico_tasa(
                ventas,
                "rango_valor",
                "Cancelación por quintiles de monto bruto"
            )

    if "shipping_cost_ratio" in filtrado.columns:

        envios = filtrado.dropna(
            subset=["shipping_cost_ratio"]
        ).copy()

        if (
            envios["shipping_cost_ratio"].nunique()
            > 1
        ):

            envios["rango_envio"] = pd.qcut(
                envios["shipping_cost_ratio"],
                q=5,
                duplicates="drop"
            )

            grafico_tasa(
                envios,
                "rango_envio",
                "Cancelación por quintiles de costo de envío"
            )

    if disponibles:

        variable = st.selectbox(
            "Variable para comparar distribuciones",
            options=[
                columna
                for columna, _ in disponibles
            ]
        )

        grafico_box(
            filtrado,
            variable,
            f"Distribución de {variable} por estado"
        )

    if "sales_channel" in filtrado.columns:

        grafico_tasa(
            filtrado,
            "sales_channel",
            "Cancelación por canal de venta",
            ordenar=True
        )

    aviso(
        "Los importes provienen de la tabla principal "
        "de ventas. Durante la auditoría se identificaron "
        "discrepancias frente al detalle de órdenes. "
        "No se consideran cifras contables verificadas."
    )


# ============================================================
# VISTA 05 - EVALUACIÓN PREDICTIVA
# ============================================================

elif codigo_pagina == "05":

    st.caption(
        "Las métricas corresponden al conjunto Test "
        "original y no cambian con los filtros descriptivos."
    )

    columnas = st.columns(5)

    indicadores = [
        ("AUC-ROC", "auc_roc", False),
        ("Accuracy", "accuracy", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-Score", "f1_score", True)
    ]

    for columna_visual, (
        nombre,
        clave,
        porcentaje
    ) in zip(
        columnas,
        indicadores
    ):

        with columna_visual:

            tarjeta(
                nombre,
                formato_metrica(
                    clave,
                    porcentaje
                ),
                "Test temporal"
            )

    st.write("")

    # --------------------------------------------------------
    # MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    if confusion is not None:

        if {
            "resultado",
            "cantidad"
        }.issubset(confusion.columns):

            valores = dict(
                zip(
                    confusion["resultado"],
                    confusion["cantidad"]
                )
            )

            claves = [
                "TN",
                "FP",
                "FN",
                "TP"
            ]

            if all(
                clave in valores
                for clave in claves
            ):

                matriz = np.array([
                    [
                        valores["TN"],
                        valores["FP"]
                    ],
                    [
                        valores["FN"],
                        valores["TP"]
                    ]
                ])

                fig = go.Figure(
                    go.Heatmap(
                        z=matriz,
                        x=[
                            "Pred. Completed",
                            "Pred. Cancelled"
                        ],
                        y=[
                            "Real Completed",
                            "Real Cancelled"
                        ],
                        text=matriz,
                        texttemplate="%{text:,}",
                        colorscale=[
                            [0, BG],
                            [0.5, BLUE],
                            [1, CYAN]
                        ]
                    )
                )

                mostrar(
                    tema(
                        fig,
                        "Matriz de confusión | Test"
                    )
                )

    # --------------------------------------------------------
    # COMPARACIÓN CON MODELO REFERENCIAL
    # --------------------------------------------------------

    if comparacion is not None:

        if {
            "modelo",
            "auc_promedio"
        }.issubset(comparacion.columns):

            fig = px.bar(
                comparacion,
                x="modelo",
                y="auc_promedio",
                color="modelo",
                color_discrete_sequence=[
                    CYAN,
                    BLUE
                ],
                text="auc_promedio"
            )

            fig.update_traces(
                texttemplate="%{text:.4f}"
            )

            mostrar(
                tema(
                    fig,
                    "AUC promedio | Validación temporal"
                )
            )

    # --------------------------------------------------------
    # EVALUACIÓN DE UMBRALES
    # --------------------------------------------------------

    if umbrales is not None:

        requeridas = {
            "umbral",
            "precision",
            "recall",
            "f1_score"
        }

        if requeridas.issubset(
            umbrales.columns
        ):

            fig = go.Figure()

            configuracion = [
                ("precision", "Precision", CYAN),
                ("recall", "Recall", BLUE),
                ("f1_score", "F1-Score", GREEN)
            ]

            for variable, nombre, color in configuracion:

                fig.add_trace(
                    go.Scatter(
                        x=umbrales["umbral"],
                        y=umbrales[variable],
                        mode="lines+markers",
                        name=nombre,
                        line=dict(
                            color=color,
                            width=3
                        )
                    )
                )

            mostrar(
                tema(
                    fig,
                    "Precision, Recall y F1 por umbral | Validación"
                )
            )

    # --------------------------------------------------------
    # COEFICIENTES DEL MODELO
    # --------------------------------------------------------

    if coeficientes is not None:

        requeridas = {
            "variable",
            "coeficiente_estandarizado"
        }

        if requeridas.issubset(
            coeficientes.columns
        ):

            fig = px.bar(
                coeficientes,
                x="variable",
                y="coeficiente_estandarizado",
                color_discrete_sequence=[TURQUOISE]
            )

            mostrar(
                tema(
                    fig,
                    "Coeficientes estandarizados del modelo"
                )
            )

    # --------------------------------------------------------
    # UMBRAL SELECCIONADO
    # --------------------------------------------------------

    if (
        umbral_seleccionado is not None
        and "umbral_seleccionado"
        in umbral_seleccionado.columns
    ):

        mejor_umbral = float(
            umbral_seleccionado.iloc[0][
                "umbral_seleccionado"
            ]
        )

        aviso(
            f"Umbral seleccionado mediante validación "
            f"temporal: {mejor_umbral:.2f}. "
            "Criterio utilizado: maximización del F1-Score."
        )

    st.warning(
        "El modelo presenta capacidad predictiva limitada. "
        "No existe evidencia suficiente para recomendar "
        "su implementación comercial."
    )


# ============================================================
# VISTA 06 - EXPLORADOR DE RIESGO
# ============================================================

elif codigo_pagina == "06":

    if predicciones is None:

        st.warning(
            "No se encontró 44_predicciones_test.csv"
        )

    else:

        pred = predicciones.copy()

        columnas_necesarias = {
            "order_id",
            "target",
            "probabilidad_cancelacion"
        }

        if not columnas_necesarias.issubset(
            pred.columns
        ):

            st.error(
                "El archivo de predicciones no contiene "
                "las columnas necesarias."
            )

        else:

            pred["target"] = pd.to_numeric(
                pred["target"],
                errors="coerce"
            )

            pred["probabilidad_cancelacion"] = (
                pd.to_numeric(
                    pred["probabilidad_cancelacion"],
                    errors="coerce"
                )
            )

            pred = pred.dropna(
                subset=[
                    "target",
                    "probabilidad_cancelacion"
                ]
            ).copy()

            pred["target"] = (
                pred["target"].astype(int)
            )

            pred["Estado real"] = (
                pred["target"].map({
                    0: "Completed",
                    1: "Cancelled"
                })
            )

            if pred.empty:

                st.info(
                    "No existen predicciones válidas."
                )

            else:

                # --------------------------------------------
                # KPIs DEL CONJUNTO TEST
                # --------------------------------------------

                c1, c2, c3 = st.columns(3)

                with c1:

                    tarjeta(
                        "Órdenes Test",
                        f"{len(pred):,}",
                        "Observaciones evaluadas"
                    )

                with c2:

                    tarjeta(
                        "Mediana de probabilidad",
                        f"{pred['probabilidad_cancelacion'].median():.2%}",
                        "Probabilidades estimadas"
                    )

                with c3:

                    tarjeta(
                        "Cancelaciones reales",
                        f"{int(pred['target'].sum()):,}",
                        "Target = 1"
                    )

                st.write("")

                # --------------------------------------------
                # DISTRIBUCIÓN DE PROBABILIDADES
                # --------------------------------------------

                fig = px.histogram(
                    pred,
                    x="probabilidad_cancelacion",
                    color="Estado real",
                    color_discrete_map=STATUS_COLORS,
                    nbins=40,
                    barmode="overlay",
                    histnorm="probability density",
                    opacity=0.75
                )

                mostrar(
                    tema(
                        fig,
                        "Distribución de probabilidades por estado"
                    )
                )

                # --------------------------------------------
                # SIMULACIÓN DE UMBRAL
                # --------------------------------------------

                st.subheader(
                    "Simulación visual de umbral"
                )

                umbral = st.slider(
                    "Umbral de clasificación",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.01
                )

                pred["clasificacion_visual"] = (
                    pred["probabilidad_cancelacion"]
                    >= umbral
                ).astype(int)

                y_real = pred["target"]
                y_pred = pred["clasificacion_visual"]

                tp = int(
                    (
                        (y_real == 1)
                        & (y_pred == 1)
                    ).sum()
                )

                fp = int(
                    (
                        (y_real == 0)
                        & (y_pred == 1)
                    ).sum()
                )

                fn = int(
                    (
                        (y_real == 1)
                        & (y_pred == 0)
                    ).sum()
                )

                tn = int(
                    (
                        (y_real == 0)
                        & (y_pred == 0)
                    ).sum()
                )

                precision = (
                    tp / (tp + fp)
                    if (tp + fp) > 0
                    else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if (tp + fn) > 0
                    else 0.0
                )

                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    tarjeta(
                        "Precision simulada",
                        f"{precision:.2%}"
                    )

                with c2:

                    tarjeta(
                        "Recall simulado",
                        f"{recall:.2%}"
                    )

                with c3:

                    tarjeta(
                        "F1 simulado",
                        f"{f1:.2%}"
                    )

                with c4:

                    tarjeta(
                        "Falsos positivos",
                        f"{fp:,}"
                    )

                # --------------------------------------------
                # MATRIZ SIMULADA
                # --------------------------------------------

                matriz = np.array([
                    [tn, fp],
                    [fn, tp]
                ])

                fig = go.Figure(
                    go.Heatmap(
                        z=matriz,
                        x=[
                            "Pred. Completed",
                            "Pred. Cancelled"
                        ],
                        y=[
                            "Real Completed",
                            "Real Cancelled"
                        ],
                        text=matriz,
                        texttemplate="%{text:,}",
                        colorscale=[
                            [0, BG],
                            [0.5, BLUE],
                            [1, CYAN]
                        ]
                    )
                )

                mostrar(
                    tema(
                        fig,
                        "Matriz de confusión | Umbral visual"
                    )
                )

                # --------------------------------------------
                # EXPLORADOR DE ÓRDENES
                # --------------------------------------------

                st.subheader(
                    "Consulta de órdenes evaluadas"
                )

                minimo = st.slider(
                    "Probabilidad mínima para mostrar",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    key="probabilidad_minima"
                )

                tabla = pred[
                    pred["probabilidad_cancelacion"]
                    >= minimo
                ].copy()

                tabla = tabla.sort_values(
                    "probabilidad_cancelacion",
                    ascending=False
                )

                st.write(
                    f"Órdenes encontradas: {len(tabla):,}"
                )

                columnas_mostrar = [
                    columna
                    for columna in [
                        "order_id",
                        "order_date",
                        "Estado real",
                        "probabilidad_cancelacion",
                        "clasificacion_visual"
                    ]
                    if columna in tabla.columns
                ]

                st.dataframe(
                    tabla[columnas_mostrar].head(1000),
                    use_container_width=True,
                    hide_index=True
                )

                st.warning(
                    "La simulación de umbrales sobre Test "
                    "tiene fines exploratorios. No debe "
                    "utilizarse para seleccionar un nuevo "
                    "umbral ni presentar una mejora "
                    "predictiva validada."
                )


# ============================================================
# 15. PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "TAF | Business Data & AI Strategy | "
    "E-Commerce Cancellation Analytics | "
    "Metodología CRISP-DM | "
    "Asociación no implica causalidad."
)
