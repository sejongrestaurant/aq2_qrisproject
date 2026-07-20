import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { AppLayout } from "./components/AppLayout";
import { Backtest } from "./pages/Backtest";
import { CurrentPortfolio } from "./pages/CurrentPortfolio";
import { FactorAnalysis } from "./pages/FactorAnalysis";
import { MarketRegime } from "./pages/MarketRegime";
import { Methodology } from "./pages/Methodology";
import { Overview } from "./pages/Overview";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1
    }
  }
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <Overview /> },
      { path: "portfolio", element: <CurrentPortfolio /> },
      { path: "regime", element: <MarketRegime /> },
      { path: "backtest", element: <Backtest /> },
      { path: "factors", element: <FactorAnalysis /> },
      { path: "methodology", element: <Methodology /> }
    ]
  }
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>
);
