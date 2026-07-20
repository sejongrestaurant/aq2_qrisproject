import type { ReactNode } from "react";

interface StateProps {
  title: string;
  description?: string;
}

export function LoadingState({ title = "데이터를 불러오는 중입니다" }: Partial<StateProps>): ReactNode {
  return <div className="state state-loading">{title}</div>;
}

export function ErrorState({ title, description }: StateProps): ReactNode {
  return (
    <div className="state state-error">
      <strong>{title}</strong>
      {description ? <span>{description}</span> : null}
    </div>
  );
}

export function EmptyState({ title, description }: StateProps): ReactNode {
  return (
    <div className="state">
      <strong>{title}</strong>
      {description ? <span>{description}</span> : null}
    </div>
  );
}
