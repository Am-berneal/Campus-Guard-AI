import PageHeader from '../components/PageHeader.jsx';
import EmptyState from '../components/EmptyState.jsx';

export default function IncidentsPage() {
  return (
    <>
      <PageHeader
        title="Incidents"
        subtitle="Possible safety events flagged for administrator review."
      />

      <section className="panel">
        <EmptyState icon="⚑" title="No incidents to show">
          <p>Incident management is not connected yet.</p>
          <p>
            In Phase 2 this page will list incidents from the backend, starting with clearly labelled
            <strong> TEST</strong> incidents to verify the workflow.
          </p>
        </EmptyState>
      </section>
    </>
  );
}
