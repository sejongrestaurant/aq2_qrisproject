import type { ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/portfolio", label: "Current Portfolio" },
  { to: "/regime", label: "Market Regime" },
  { to: "/backtest", label: "Backtest" },
  { to: "/factors", label: "Factor Analysis" },
  { to: "/methodology", label: "Methodology" }
];

export function AppLayout(): ReactNode {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <strong>MUST-30</strong>
          <span>Active Strategy</span>
        </div>
        <nav>
          {navItems.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === "/"}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <p className="disclaimer">
          본 대시보드는 교육 및 프로젝트 검증 목적입니다. 투자 권유나 매매 추천이 아닙니다.
        </p>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
