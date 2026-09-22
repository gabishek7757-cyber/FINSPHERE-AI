import React, { useState, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend
} from "recharts";
import {
  LayoutDashboard, PieChart, ClipboardCheck, SlidersHorizontal,
  UserCircle2, Settings, ShieldAlert, ShieldCheck, TrendingUp, Target
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* Design tokens — pulled from the FinSphere AI brief                  */
/* ------------------------------------------------------------------ */
const COLORS = {
  bg: "#F8FAFC",
  navy: "#0F172A",
  navySoft: "#1E293B",
  teal: "#14B8A6",
  tealSoft: "#CCFBF1",
  coral: "#EF4444",
  coralSoft: "#FEE2E2",
  ink: "#0F172A",
  inkSoft: "#64748B",
  card: "#FFFFFF",
  border: "#E2E8F0",
};

/* ------------------------------------------------------------------ */
/* Illustrative data                                                   */
/* Notebook uploaded (fraud_detection_improved.ipynb) has no saved      */
/* execution outputs — fraudTrain.csv / fraudTest.csv were never run.   */
/* Figures below are placeholders shaped like real model output; swap   */
/* in your notebook's printed numbers once you re-run it.               */
/* ------------------------------------------------------------------ */
const kpis = [
  { label: "Total transactions analyzed", value: "1,293,485", icon: LayoutDashboard, tone: "teal" },
  { label: "Overall fraud rate", value: "0.57%", icon: ShieldAlert, tone: "coral", note: "Highly imbalanced" },
  { label: "Random Forest F1 score", value: "0.8241", icon: Target, tone: "teal" },
  { label: "Optimal decision threshold", value: "0.35", icon: SlidersHorizontal, tone: "teal" },
];

const fraudByCategory = [
  { category: "grocery_pos", rate: 4.0 },
  { category: "shopping_net", rate: 3.1 },
  { category: "misc_net", rate: 2.8 },
  { category: "travel", rate: 2.5 },
  { category: "shopping_pos", rate: 1.9 },
  { category: "gas_transport", rate: 1.5 },
  { category: "entertainment", rate: 0.9 },
  { category: "personal_care", rate: 0.5 },
].sort((a, b) => b.rate - a.rate);

const fraudByHour = Array.from({ length: 24 }, (_, h) => {
  // late-night hours (22-03) carry elevated fraud rates, matching notebook EDA
  const nightBump = h >= 22 || h <= 3 ? 5.5 : 0;
  const base = 1.2 + Math.sin((h / 24) * Math.PI * 2) * 0.4;
  return { hour: h, rate: Math.max(0.3, +(base + nightBump).toFixed(2)) };
});

const featureImportance = [
  { feature: "distance_km", importance: 0.31 },
  { feature: "amt_log", importance: 0.24 },
  { feature: "hour", importance: 0.17 },
  { feature: "category", importance: 0.12 },
  { feature: "age", importance: 0.08 },
  { feature: "day_of_week", importance: 0.05 },
  { feature: "gender", importance: 0.03 },
].sort((a, b) => a.importance - b.importance);

const confusionMatrices = {
  "Logistic Regression": { tn: 550676, fp: 1876, fn: 793, tp: 2374, precision: 0.559, recall: 0.750, f1: 0.640 },
  "Random Forest": { tn: 552023, fp: 529, fn: 566, tp: 2601, precision: 0.831, recall: 0.821, f1: 0.826 },
};

const NAV_ITEMS = [
  { key: "overview", label: "Overview", icon: LayoutDashboard },
  { key: "eda", label: "EDA", icon: PieChart },
  { key: "evaluation", label: "Evaluation", icon: ClipboardCheck },
  { key: "tuning", label: "Tuning", icon: SlidersHorizontal },
];

/* ------------------------------------------------------------------ */
/* Small building blocks                                               */
/* ------------------------------------------------------------------ */
function Card({ children, style }) {
  return (
    <div
      style={{
        background: COLORS.card,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 16,
        boxShadow: "0 1px 3px rgba(15,23,42,0.06)",
        padding: 20,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{ fontSize: 15, fontWeight: 600, color: COLORS.ink, margin: "0 0 14px 0", letterSpacing: 0.2 }}>
      {children}
    </h2>
  );
}

function KpiCard({ kpi }) {
  const Icon = kpi.icon;
  const accent = kpi.tone === "coral" ? COLORS.coral : COLORS.teal;
  const accentBg = kpi.tone === "coral" ? COLORS.coralSoft : COLORS.tealSoft;
  return (
    <Card style={{ flex: "1 1 200px", minWidth: 190 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ fontSize: 12.5, color: COLORS.inkSoft, fontWeight: 500 }}>{kpi.label}</span>
        <div
          style={{
            width: 30, height: 30, borderRadius: 9, background: accentBg,
            display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
          }}
        >
          <Icon size={16} color={accent} strokeWidth={2.25} />
        </div>
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color: COLORS.ink, lineHeight: 1.1 }}>{kpi.value}</div>
      {kpi.note && (
        <div style={{ fontSize: 11.5, color: accent, fontWeight: 500, marginTop: 6 }}>{kpi.note}</div>
      )}
    </Card>
  );
}

function ConfusionMatrix({ name, data, tone }) {
  const accent = tone === "coral" ? COLORS.coral : COLORS.teal;
  const accentBg = tone === "coral" ? COLORS.coralSoft : COLORS.tealSoft;
  const cell = (label, value, emphasize) => (
    <div
      style={{
        background: emphasize ? accentBg : "#F1F5F9",
        borderRadius: 10,
        padding: "12px 8px",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 10.5, color: COLORS.inkSoft, fontWeight: 600, letterSpacing: 0.3, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, color: emphasize ? accent : COLORS.ink }}>
        {value.toLocaleString()}
      </div>
    </div>
  );
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.ink, marginBottom: 10 }}>{name}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {cell("True negative", data.tn, false)}
        {cell("False positive", data.fp, true)}
        {cell("False negative", data.fn, true)}
        {cell("True positive", data.tp, false)}
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 12, color: COLORS.inkSoft }}>
        <span>Precision <b style={{ color: COLORS.ink }}>{data.precision.toFixed(3)}</b></span>
        <span>Recall <b style={{ color: COLORS.ink }}>{data.recall.toFixed(3)}</b></span>
        <span>F1 <b style={{ color: COLORS.ink }}>{data.f1.toFixed(3)}</b></span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main dashboard                                                       */
/* ------------------------------------------------------------------ */
export default function FinSphereDashboard() {
  const [active, setActive] = useState("overview");
  const refs = {
    overview: useRef(null),
    eda: useRef(null),
    evaluation: useRef(null),
    tuning: useRef(null),
  };

  const scrollTo = (key) => {
    setActive(key);
    refs[key].current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: COLORS.bg, fontFamily: "'Inter', system-ui, sans-serif" }}>
      {/* Sidebar */}
      <aside style={{ width: 220, background: COLORS.navy, flexShrink: 0, padding: "24px 14px", position: "sticky", top: 0, height: "100vh" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 10px 26px 10px" }}>
          <ShieldCheck size={22} color={COLORS.teal} strokeWidth={2.25} />
          <span style={{ color: "#F1F5F9", fontWeight: 700, fontSize: 15 }}>FinSphere AI</span>
        </div>
        <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = active === item.key;
            return (
              <button
                key={item.key}
                onClick={() => scrollTo(item.key)}
                style={{
                  display: "flex", alignItems: "center", gap: 10,
                  background: isActive ? "rgba(20,184,166,0.14)" : "transparent",
                  color: isActive ? COLORS.teal : "#94A3B8",
                  border: "none", borderRadius: 10, padding: "9px 12px",
                  fontSize: 13.5, fontWeight: 500, cursor: "pointer", textAlign: "left",
                  borderLeft: isActive ? `3px solid ${COLORS.teal}` : "3px solid transparent",
                  transition: "all 0.15s ease",
                }}
              >
                <Icon size={16} strokeWidth={2} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </aside>

      {/* Main content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Header */}
        <header
          style={{
            background: COLORS.navy, padding: "16px 28px",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <div>
            <div style={{ color: "#F1F5F9", fontWeight: 700, fontSize: 16 }}>FinSphere AI</div>
            <div style={{ color: "#94A3B8", fontSize: 12 }}>Fraud detection module</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <button style={{ background: "none", border: "none", cursor: "pointer" }} aria-label="Settings">
              <Settings size={18} color="#94A3B8" />
            </button>
            <button style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }} aria-label="User profile">
              <UserCircle2 size={22} color="#CBD5E1" />
            </button>
          </div>
        </header>

        <main style={{ padding: 28, display: "flex", flexDirection: "column", gap: 28, maxWidth: 1180 }}>
          {/* Overview / KPIs */}
          <section ref={refs.overview}>
            <SectionTitle>Summary metrics</SectionTitle>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              {kpis.map((k) => <KpiCard key={k.label} kpi={k} />)}
            </div>
          </section>

          {/* EDA */}
          <section ref={refs.eda}>
            <SectionTitle>Exploratory data analysis</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <Card>
                <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.ink, marginBottom: 12 }}>
                  Fraud rate by merchant category
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={fraudByCategory} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 11, fill: COLORS.inkSoft }} unit="%" />
                    <YAxis type="category" dataKey="category" width={100} tick={{ fontSize: 11, fill: COLORS.inkSoft }} />
                    <Tooltip formatter={(v) => [`${v}%`, "Fraud rate"]} contentStyle={{ fontSize: 12, borderRadius: 8, border: `1px solid ${COLORS.border}` }} />
                    <Bar dataKey="rate" fill={COLORS.coral} radius={[0, 6, 6, 0]} barSize={14} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.ink, marginBottom: 12 }}>
                  Fraud rate by hour of day
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={fraudByHour} margin={{ left: 4, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} vertical={false} />
                    <XAxis dataKey="hour" tick={{ fontSize: 11, fill: COLORS.inkSoft }} interval={2} />
                    <YAxis tick={{ fontSize: 11, fill: COLORS.inkSoft }} unit="%" />
                    <Tooltip formatter={(v) => [`${v}%`, "Fraud rate"]} labelFormatter={(h) => `${h}:00`} contentStyle={{ fontSize: 12, borderRadius: 8, border: `1px solid ${COLORS.border}` }} />
                    <Line type="monotone" dataKey="rate" stroke={COLORS.teal} strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </section>

          {/* Evaluation */}
          <section ref={refs.evaluation}>
            <SectionTitle>Model evaluation — Random Forest vs. Logistic Regression</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.15fr", gap: 16 }}>
              <Card>
                <ConfusionMatrix name="Logistic Regression" data={confusionMatrices["Logistic Regression"]} tone="coral" />
              </Card>
              <Card>
                <ConfusionMatrix name="Random Forest" data={confusionMatrices["Random Forest"]} tone="teal" />
              </Card>
              <Card>
                <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.ink, marginBottom: 12 }}>
                  Feature importance (Random Forest)
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={featureImportance} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 11, fill: COLORS.inkSoft }} />
                    <YAxis type="category" dataKey="feature" width={90} tick={{ fontSize: 11, fill: COLORS.inkSoft }} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: `1px solid ${COLORS.border}` }} />
                    <Bar dataKey="importance" fill={COLORS.teal} radius={[0, 6, 6, 0]} barSize={14} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </section>

          {/* Tuning */}
          <section ref={refs.tuning} style={{ paddingBottom: 40 }}>
            <SectionTitle>Threshold tuning</SectionTitle>
            <Card>
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, background: COLORS.tealSoft,
                  display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                }}>
                  <TrendingUp size={20} color={COLORS.teal} strokeWidth={2.25} />
                </div>
                <div>
                  <div style={{ fontSize: 13.5, color: COLORS.ink, fontWeight: 600 }}>
                    Optimal decision threshold: 0.35
                  </div>
                  <div style={{ fontSize: 12.5, color: COLORS.inkSoft, marginTop: 2 }}>
                    Scanning thresholds from 0.10–0.90 and maximizing F1 for the fraud class lifts F1
                    above the default 0.50 cutoff — recommended for production scoring.
                  </div>
                </div>
              </div>
            </Card>
          </section>
        </main>
      </div>
    </div>
  );
}
