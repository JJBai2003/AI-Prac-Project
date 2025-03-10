import { Link } from "react-router-dom";

export default function NavBar() {
    return (
        <div className="nav">
            <Link to="/" className="nav-home">Chompions</Link>
            <Link to="/faq" className="nav-link">Faq</Link>
        </div>
    );
}
