import { Link } from 'react-router';
import EmptyState from '../components/EmptyState.jsx';

export default function NotFoundPage() {
  return (
    <section className="panel">
      <EmptyState icon="?" title="Page not found">
        <p>
          <Link to="/dashboard">Go to the dashboard</Link>
        </p>
      </EmptyState>
    </section>
  );
}
